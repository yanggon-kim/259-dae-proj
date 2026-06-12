#include "wasp-rfq.h"

#include <assert.h>
#include <algorithm>

wasp_rfq::wasp_rfq()
    : m_entries(0),
      m_payload_words_per_lane(0),
      m_warp_size(0),
      m_head(0),
      m_tail(0),
      m_occupancy(0),
      m_next_sequence(1) {}

void wasp_rfq::configure(unsigned entries, unsigned payload_words_per_lane,
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

bool wasp_rfq::head_ready() const {
  if (empty()) return false;
  return m_storage[m_head].state == RFQ_READY;
}

unsigned wasp_rfq::ready_count() const {
  unsigned count = 0;
  for (unsigned i = 0; i < m_entries; ++i)
    if (m_storage[i].state == RFQ_READY) count++;
  return count;
}

unsigned wasp_rfq::pending_count() const {
  unsigned count = 0;
  for (unsigned i = 0; i < m_entries; ++i)
    if (m_storage[i].state == RFQ_RESERVED) count++;
  return count;
}

wasp_rfq_token wasp_rfq::reserve_tail(const wasp_rfq_key &key,
                                      const active_mask_t &active_mask,
                                      unsigned pending_responses) {
  assert(configured());
  assert(!full());
  entry &e = m_storage[m_tail];
  assert(e.state == RFQ_FREE);

  e.state = pending_responses == 0 ? RFQ_READY : RFQ_RESERVED;
  e.sequence = m_next_sequence++;
  e.pending_responses = pending_responses;
  e.payload_words_per_lane = m_payload_words_per_lane;
  e.active_mask = active_mask;
  std::fill(e.lane_payload.begin(), e.lane_payload.end(), 0);

  wasp_rfq_token token;
  token.key = key;
  token.slot_index = m_tail;
  token.sequence = e.sequence;
  token.payload_words_per_lane = m_payload_words_per_lane;
  token.valid = true;

  m_tail = (m_tail + 1) % m_entries;
  m_occupancy++;
  return token;
}

void wasp_rfq::set_pending_responses(const wasp_rfq_token &token,
                                     unsigned pending_responses) {
  entry &e = entry_for_token(token);
  e.pending_responses = pending_responses;
  e.state = pending_responses == 0 ? RFQ_READY : RFQ_RESERVED;
}

bool wasp_rfq::has_pending_responses(const wasp_rfq_token &token) const {
  const entry &e = entry_for_token(token);
  return e.pending_responses > 0;
}

bool wasp_rfq::mark_response(const wasp_rfq_token &token) {
  entry &e = entry_for_token(token);
  assert(e.state == RFQ_RESERVED || e.state == RFQ_READY);
  assert(e.pending_responses > 0);
  e.pending_responses--;
  if (e.pending_responses == 0) {
    e.state = RFQ_READY;
    return true;
  }
  return false;
}

void wasp_rfq::consume_head(const wasp_rfq_key &key) {
  (void)key;
  assert(!empty());
  entry &e = m_storage[m_head];
  assert(e.state == RFQ_READY);
  e.state = RFQ_FREE;
  e.pending_responses = 0;
  e.active_mask.reset();
  std::fill(e.lane_payload.begin(), e.lane_payload.end(), 0);
  m_head = (m_head + 1) % m_entries;
  m_occupancy--;
}

void wasp_rfq::set_lane_value(const wasp_rfq_token &token, unsigned lane,
                              unsigned word, uint64_t value) {
  entry &e = entry_for_token(token);
  assert(lane < m_warp_size);
  assert(word < e.payload_words_per_lane);
  e.lane_payload[lane * e.payload_words_per_lane + word] = value;
}

uint64_t wasp_rfq::head_lane_value(unsigned lane, unsigned word) const {
  assert(head_ready());
  const entry &e = m_storage[m_head];
  assert(lane < m_warp_size);
  assert(word < e.payload_words_per_lane);
  return e.lane_payload[lane * e.payload_words_per_lane + word];
}

wasp_rfq::entry &wasp_rfq::entry_for_token(const wasp_rfq_token &token) {
  assert(token.valid);
  assert(token.slot_index < m_entries);
  entry &e = m_storage[token.slot_index];
  assert(e.sequence == token.sequence);
  return e;
}

const wasp_rfq::entry &wasp_rfq::entry_for_token(
    const wasp_rfq_token &token) const {
  assert(token.valid);
  assert(token.slot_index < m_entries);
  const entry &e = m_storage[token.slot_index];
  assert(e.sequence == token.sequence);
  return e;
}

wasp_rfq &wasp_rfq_table::get_or_create(const wasp_rfq_key &key,
                                        unsigned entries,
                                        unsigned payload_words_per_lane,
                                        unsigned warp_size) {
  wasp_rfq &queue = m_queues[key];
  queue.configure(entries, payload_words_per_lane, warp_size);
  return queue;
}

wasp_rfq *wasp_rfq_table::find(const wasp_rfq_key &key) {
  std::map<wasp_rfq_key, wasp_rfq>::iterator it = m_queues.find(key);
  return it == m_queues.end() ? NULL : &it->second;
}

const wasp_rfq *wasp_rfq_table::find(const wasp_rfq_key &key) const {
  std::map<wasp_rfq_key, wasp_rfq>::const_iterator it = m_queues.find(key);
  return it == m_queues.end() ? NULL : &it->second;
}

void wasp_rfq_table::erase_cta(unsigned cta_hw_id) {
  std::map<wasp_rfq_key, wasp_rfq>::iterator it = m_queues.begin();
  while (it != m_queues.end()) {
    if (it->first.cta_hw_id == cta_hw_id)
      it = m_queues.erase(it);
    else
      ++it;
  }
}
