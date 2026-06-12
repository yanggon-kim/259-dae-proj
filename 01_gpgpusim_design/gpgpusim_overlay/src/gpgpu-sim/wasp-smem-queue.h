#ifndef WASP_SMEM_QUEUE_H_
#define WASP_SMEM_QUEUE_H_

#include <stdint.h>

#include <map>
#include <vector>

#include "../abstract_hardware_model.h"

struct smemq_key {
  unsigned cta_hw_id;
  unsigned original_warp_id;
  unsigned src_stage_id;
  unsigned dst_stage_id;
  unsigned queue_id;

  smemq_key()
      : cta_hw_id(0),
        original_warp_id(0),
        src_stage_id(0),
        dst_stage_id(1),
        queue_id(0) {}

  bool operator<(const smemq_key &rhs) const {
    if (cta_hw_id != rhs.cta_hw_id) return cta_hw_id < rhs.cta_hw_id;
    if (original_warp_id != rhs.original_warp_id)
      return original_warp_id < rhs.original_warp_id;
    if (src_stage_id != rhs.src_stage_id)
      return src_stage_id < rhs.src_stage_id;
    if (dst_stage_id != rhs.dst_stage_id)
      return dst_stage_id < rhs.dst_stage_id;
    return queue_id < rhs.queue_id;
  }
};

struct smemq_token {
  smemq_key key;
  unsigned slot_index;
  unsigned sequence;
  unsigned payload_words_per_lane;
  bool valid;

  smemq_token()
      : slot_index(0),
        sequence(0),
        payload_words_per_lane(0),
        valid(false) {}
};

class smemq {
 public:
  smemq();

  void configure(unsigned entries, unsigned payload_words_per_lane,
                 unsigned warp_size);
  bool configured() const { return m_entries > 0; }
  bool full() const { return m_occupancy >= m_entries; }
  bool empty() const { return m_occupancy == 0; }
  bool head_ready() const;
  unsigned occupancy() const { return m_occupancy; }
  unsigned ready_count() const;
  unsigned pending_count() const;

  smemq_token reserve_tail(const smemq_key &key,
                           const active_mask_t &active_mask,
                           unsigned pending_responses);
  void set_pending_responses(const smemq_token &token,
                             unsigned pending_responses);
  bool has_pending_responses(const smemq_token &token) const;
  bool mark_response(const smemq_token &token);
  void consume_head(const smemq_key &key);

  void set_lane_value(const smemq_token &token, unsigned lane, unsigned word,
                      uint64_t value);
  uint64_t head_lane_value(unsigned lane, unsigned word) const;

 private:
  enum entry_state { SMEMQ_FREE = 0, SMEMQ_RESERVED, SMEMQ_READY };

  struct entry {
    entry()
        : state(SMEMQ_FREE),
          sequence(0),
          pending_responses(0),
          payload_words_per_lane(0) {}

    entry_state state;
    unsigned sequence;
    unsigned pending_responses;
    unsigned payload_words_per_lane;
    active_mask_t active_mask;
    std::vector<uint64_t> lane_payload;
  };

  entry &entry_for_token(const smemq_token &token);
  const entry &entry_for_token(const smemq_token &token) const;

  unsigned m_entries;
  unsigned m_payload_words_per_lane;
  unsigned m_warp_size;
  unsigned m_head;
  unsigned m_tail;
  unsigned m_occupancy;
  unsigned m_next_sequence;
  std::vector<entry> m_storage;
};

class smemq_table {
 public:
  smemq &get_or_create(const smemq_key &key, unsigned entries,
                       unsigned payload_words_per_lane, unsigned warp_size);
  smemq *find(const smemq_key &key);
  const smemq *find(const smemq_key &key) const;
  void erase_cta(unsigned cta_hw_id);
  void clear() { m_queues.clear(); }

 private:
  std::map<smemq_key, smemq> m_queues;
};

#endif
