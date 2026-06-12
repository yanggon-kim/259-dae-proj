#ifndef WASP_RFQ_H_
#define WASP_RFQ_H_

#include <stdint.h>

#include <map>
#include <vector>

#include "../abstract_hardware_model.h"

struct wasp_rfq_key {
  unsigned cta_hw_id;
  unsigned original_warp_id;
  unsigned src_stage_id;
  unsigned dst_stage_id;
  unsigned queue_id;

  wasp_rfq_key()
      : cta_hw_id(0),
        original_warp_id(0),
        src_stage_id(0),
        dst_stage_id(1),
        queue_id(0) {}

  bool operator<(const wasp_rfq_key &rhs) const {
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

struct wasp_rfq_token {
  wasp_rfq_key key;
  unsigned slot_index;
  unsigned sequence;
  unsigned payload_words_per_lane;
  bool valid;

  wasp_rfq_token()
      : slot_index(0),
        sequence(0),
        payload_words_per_lane(0),
        valid(false) {}
};

class wasp_rfq {
 public:
  wasp_rfq();

  void configure(unsigned entries, unsigned payload_words_per_lane,
                 unsigned warp_size);
  bool configured() const { return m_entries > 0; }
  bool full() const { return m_occupancy >= m_entries; }
  bool empty() const { return m_occupancy == 0; }
  bool head_ready() const;
  unsigned occupancy() const { return m_occupancy; }
  unsigned ready_count() const;
  unsigned pending_count() const;

  wasp_rfq_token reserve_tail(const wasp_rfq_key &key,
                              const active_mask_t &active_mask,
                              unsigned pending_responses);
  void set_pending_responses(const wasp_rfq_token &token,
                             unsigned pending_responses);
  bool has_pending_responses(const wasp_rfq_token &token) const;
  bool mark_response(const wasp_rfq_token &token);
  void consume_head(const wasp_rfq_key &key);

  void set_lane_value(const wasp_rfq_token &token, unsigned lane,
                      unsigned word, uint64_t value);
  uint64_t head_lane_value(unsigned lane, unsigned word) const;

 private:
  enum entry_state { RFQ_FREE = 0, RFQ_RESERVED, RFQ_READY };

  struct entry {
    entry()
        : state(RFQ_FREE),
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

  entry &entry_for_token(const wasp_rfq_token &token);
  const entry &entry_for_token(const wasp_rfq_token &token) const;

  unsigned m_entries;
  unsigned m_payload_words_per_lane;
  unsigned m_warp_size;
  unsigned m_head;
  unsigned m_tail;
  unsigned m_occupancy;
  unsigned m_next_sequence;
  std::vector<entry> m_storage;
};

class wasp_rfq_table {
 public:
  wasp_rfq &get_or_create(const wasp_rfq_key &key, unsigned entries,
                          unsigned payload_words_per_lane,
                          unsigned warp_size);
  wasp_rfq *find(const wasp_rfq_key &key);
  const wasp_rfq *find(const wasp_rfq_key &key) const;
  void erase_cta(unsigned cta_hw_id);
  void clear() { m_queues.clear(); }

 private:
  std::map<wasp_rfq_key, wasp_rfq> m_queues;
};

#endif
