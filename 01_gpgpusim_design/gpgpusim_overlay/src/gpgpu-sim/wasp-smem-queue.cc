#include "wasp-smem-queue.h"

#include <assert.h>
#include <algorithm>

smemq::smemq()
    : m_entries(0),
      m_payload_words_per_lane(0),
      m_warp_size(0),
      m_head(0),
      m_tail(0),
      m_occupancy(0),
      m_next_sequence(1) {}

void smemq::configure(unsigned entries, unsigned payload_words_per_lane,
                      unsigned warp_size) {
  if (configured()) {
    assert(m_entries == entries);
    assert(m_payload_words_per_lane == payload_words_per_lane);
    assert(m_warp_size == warp_size);
    return;
  }

  assert(entries > 0);
  assert(payload_words_per_lane > 0);
  assert(warp_size > 0);
  m_entries = entries;
  m_payload_words_per_lane = payload_words_per_lane;
  m_warp_size = warp_size;
  m_storage.resize(entries);
  for (unsigned i = 0; i < entries; ++i) {
    m_storage[i].lane_payload.assign(warp_size * payload_words_per_lane, 0);
  }
}

bool smemq::head_ready() const {
  if (empty()) return false;
  return m_storage[m_head].state == SMEMQ_READY;
}

unsigned smemq::ready_count() const {
  unsigned count = 0;
  for (unsigned i = 0; i < m_entries; ++i)
    if (m_storage[i].state == SMEMQ_READY) count++;
  return count;
}

unsigned smemq::pending_count() const {
  unsigned count = 0;
  for (unsigned i = 0; i < m_entries; ++i)
    if (m_storage[i].state == SMEMQ_RESERVED) count++;
  return count;
}

smemq_token smemq::reserve_tail(const smemq_key &key,
                                const active_mask_t &active_mask,
                                unsigned pending_responses) {
  assert(configured());
  assert(!full());
  entry &e = m_storage[m_tail];
  assert(e.state == SMEMQ_FREE);

  e.state = pending_responses == 0 ? SMEMQ_READY : SMEMQ_RESERVED;
  e.sequence = m_next_sequence++;
  e.pending_responses = pending_responses;
  e.payload_words_per_lane = m_payload_words_per_lane;
  e.active_mask = active_mask;
  std::fill(e.lane_payload.begin(), e.lane_payload.end(), 0);

  smemq_token token;
  token.key = key;
  token.slot_index = m_tail;
  token.sequence = e.sequence;
  token.payload_words_per_lane = m_payload_words_per_lane;
  token.valid = true;

  m_tail = (m_tail + 1) % m_entries;
  m_occupancy++;
  return token;
}

void smemq::set_pending_responses(const smemq_token &token,
                                  unsigned pending_responses) {
  entry &e = entry_for_token(token);
  e.pending_responses = pending_responses;
  e.state = pending_responses == 0 ? SMEMQ_READY : SMEMQ_RESERVED;
}

bool smemq::has_pending_responses(const smemq_token &token) const {
  const entry &e = entry_for_token(token);
  return e.pending_responses > 0;
}

bool smemq::mark_response(const smemq_token &token) {
  entry &e = entry_for_token(token);
  assert(e.state == SMEMQ_RESERVED || e.state == SMEMQ_READY);
  assert(e.pending_responses > 0);
  e.pending_responses--;
  if (e.pending_responses == 0) {
    e.state = SMEMQ_READY;
    return true;
  }
  return false;
}

void smemq::consume_head(const smemq_key &key) {
  (void)key;
  assert(!empty());
  entry &e = m_storage[m_head];
  assert(e.state == SMEMQ_READY);
  e.state = SMEMQ_FREE;
  e.pending_responses = 0;
  e.active_mask.reset();
  std::fill(e.lane_payload.begin(), e.lane_payload.end(), 0);
  m_head = (m_head + 1) % m_entries;
  m_occupancy--;
}

void smemq::set_lane_value(const smemq_token &token, unsigned lane,
                           unsigned word, uint64_t value) {
  entry &e = entry_for_token(token);
  assert(lane < m_warp_size);
  assert(word < e.payload_words_per_lane);
  e.lane_payload[lane * e.payload_words_per_lane + word] = value;
}

uint64_t smemq::head_lane_value(unsigned lane, unsigned word) const {
  assert(head_ready());
  const entry &e = m_storage[m_head];
  assert(lane < m_warp_size);
  assert(word < e.payload_words_per_lane);
  return e.lane_payload[lane * e.payload_words_per_lane + word];
}

smemq::entry &smemq::entry_for_token(const smemq_token &token) {
  assert(token.valid);
  assert(token.slot_index < m_entries);
  entry &e = m_storage[token.slot_index];
  assert(e.sequence == token.sequence);
  return e;
}

const smemq::entry &smemq::entry_for_token(const smemq_token &token) const {
  assert(token.valid);
  assert(token.slot_index < m_entries);
  const entry &e = m_storage[token.slot_index];
  assert(e.sequence == token.sequence);
  return e;
}

smemq &smemq_table::get_or_create(const smemq_key &key, unsigned entries,
                                  unsigned payload_words_per_lane,
                                  unsigned warp_size) {
  smemq &queue = m_queues[key];
  queue.configure(entries, payload_words_per_lane, warp_size);
  return queue;
}

smemq *smemq_table::find(const smemq_key &key) {
  std::map<smemq_key, smemq>::iterator it = m_queues.find(key);
  return it == m_queues.end() ? NULL : &it->second;
}

const smemq *smemq_table::find(const smemq_key &key) const {
  std::map<smemq_key, smemq>::const_iterator it = m_queues.find(key);
  return it == m_queues.end() ? NULL : &it->second;
}

void smemq_table::erase_cta(unsigned cta_hw_id) {
  std::map<smemq_key, smemq>::iterator it = m_queues.begin();
  while (it != m_queues.end()) {
    if (it->first.cta_hw_id == cta_hw_id)
      it = m_queues.erase(it);
    else
      ++it;
  }
}
