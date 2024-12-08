#ifndef MY_MEMORY_PLANNER_H_
#define MY_MEMORY_PLANNER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"

#include <map>
#include <unordered_set>
#include <cstddef>
#include <utility>

namespace tflite {
  
constexpr int kOnlinePlannedBuffer = -1;

class MyMemoryPlanner : public MicroMemoryPlanner {
 public:
  MyMemoryPlanner();
  MyMemoryPlanner(const int conflict_data[][3], const int conflict_data_count);
  ~MyMemoryPlanner() override;
  TfLiteStatus Init(unsigned char* scratch_buffer, int scratch_buffer_size);
  TfLiteStatus AddBuffer(int size, int first_time_used, int last_time_used);
  TfLiteStatus AddBuffer(int size, int first_time_used, int last_time_used, int offline_offset);
  void AddConflict(int input_tensor_id, int output_tensor_id, int l);
  size_t GetMaximumMemorySize();
  int GetBufferCount();
  TfLiteStatus GetOffsetForBuffer(int buffer_index, int* offset);
  void PrintMemoryPlan();
  struct ListEntry {
    int offset;
    int requirements_index;
    int next_entry_index;
  };
  static size_t per_buffer_size() {
    const int per_buffer_size =
      sizeof(BufferRequirements) +  // requirements_
      sizeof(int) +                 // buffer_sizes_sorted_
      sizeof(int) +                 // buffer_ids_sorted_
      sizeof(ListEntry) +           // buffers_sorted_by_offset_
      sizeof(int);                  // buffer_offsets_;
    return per_buffer_size;
  }
 private:
  void SortConflictTensorsBFS(int* buffer_ids, int* buffer_sizes, int conflict_count);
  bool DoesEntryOverlapInTime(const ListEntry* entry, const int first_time_used, const int last_time_used);
  ListEntry* NextSimultaneouslyActiveBuffer(const ListEntry* start, const int first_time_used, const int last_time_used);
  void CalculateOffsetsIfNeeded();

  int max_buffer_count_;
  int buffer_count_;

  struct BufferRequirements {
    int size;
    int first_time_used;
    int last_time_used;
    int offline_offset;
  };

  BufferRequirements* requirements_;

  std::unordered_set<int> conflict_buffer_ids_;
  std::map<std::pair<int, int>, int> conflicts_;

  int* buffer_sizes_sorted_;
  int* buffer_ids_sorted_;
  ListEntry* buffers_sorted_by_offset_;
  int next_free_entry_;

  int first_entry_index_;
  int* buffer_offsets_;
  bool need_to_calculate_offsets_;
};

} // namespace tflite

#endif  // MY_MEMORY_PLANNER_H_