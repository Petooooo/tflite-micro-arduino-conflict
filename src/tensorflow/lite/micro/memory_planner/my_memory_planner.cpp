#include "tensorflow/lite/micro/memory_planner/my_memory_planner.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_string.h"

#include <iostream>
#include <queue>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace tflite {

namespace {

// 버퍼 번호에 따른 문자열 캐릭터 반환
char GetOrdinalCharacter(int i) {
  if (i < 10) {
    return '0' + i;
  } else if (i < 36) {
    return 'a' + (i - 10);
  } else if (i < 62) {
    return 'A' + (i - 36);
  }
  return '*';
}

// 내림차순 정렬 함수 (크기 순)
void ReverseSortInPlace(int* values, int* ids, int size) {
  bool any_swapped;
  do {
    any_swapped = false;
    for (int i = 1; i < size; ++i) {
      if (values[i - 1] < values[i]) {
        const int value_temp = values[i - 1];
        values[i - 1] = values[i];
        values[i] = value_temp;
        const int id_temp = ids[i - 1];
        ids[i - 1] = ids[i];
        ids[i] = id_temp;
        any_swapped = true;
      }
    }
  } while (any_swapped);
}

} // namespace

// Conflict 정보를 바탕으로 BFS를 통해 버퍼를 정렬하는 함수
// 그래프 형태로 conflicts_ 정보를 저장하여 연쇄적 관계를 정렬
void MyMemoryPlanner::SortConflictTensorsBFS(int* buffer_ids, int* buffer_sizes, int conflict_count) {
    std::unordered_map<int, std::vector<int>> graph;
    std::unordered_set<int> roots;
    std::unordered_map<int, int> in_degree;
    std::unordered_set<int> visited; // 방문한 노드 기록

    // 그래프 구축 및 in-degree 계산
    for (const auto& conflict : conflicts_) {
        int input_id = conflict.first.first;
        int output_id = conflict.first.second;
        graph[output_id].push_back(input_id);
        in_degree[input_id]++;
    }

    // in-degree가 0인 노드를 루트로 설정
    for (const auto& conflict : conflicts_) {
        int output_id = conflict.first.second;
        if (in_degree.find(output_id) == in_degree.end()) {
            roots.insert(output_id);
        }
    }

    // BFS 수행
    std::vector<int> sorted_buffers;
    for (int root : roots) {
        std::queue<int> queue;
        queue.push(root);
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            // 이미 방문한 노드는 건너뛴다
            if (visited.find(current) != visited.end()) {
                continue;
            }
            visited.insert(current); // 방문 표시
            sorted_buffers.push_back(current);
            if (graph.find(current) != graph.end()) {
                auto& children = graph[current];
                // 자식들을 크기 순으로 정렬
                std::sort(children.begin(), children.end(),
                          [buffer_sizes, buffer_ids, conflict_count](int a, int b) {
                              auto iter_a = std::find(buffer_ids, buffer_ids + conflict_count, a);
                              auto iter_b = std::find(buffer_ids, buffer_ids + conflict_count, b);

                              int size_a = (iter_a != buffer_ids + conflict_count) ? buffer_sizes[iter_a - buffer_ids] : 0;
                              int size_b = (iter_b != buffer_ids + conflict_count) ? buffer_sizes[iter_b - buffer_ids] : 0;
                              return size_a > size_b;
                          });
                // 큐에 정렬된 자식 추가
                for (int child : children) {
                    if (visited.find(child) == visited.end()) { // 방문하지 않은 노드만 추가
                        queue.push(child);
                    }
                }
            }
        }
    }

    // BFS로 얻은 정렬 결과를 buffer_ids와 buffer_sizes에 반영
    for (int i = 0; i < conflict_count; ++i) {
      buffer_ids[i] = sorted_buffers[i];
      buffer_sizes[i] = requirements_[sorted_buffers[i]].size;
    }
}

// 생성자
MyMemoryPlanner::MyMemoryPlanner()
    : requirements_(nullptr),
      buffers_sorted_by_offset_(nullptr),
      buffer_sizes_sorted_(nullptr),
      buffer_ids_sorted_(nullptr),
      buffer_offsets_(nullptr),
      max_buffer_count_(0),
      buffer_count_(0),
      need_to_calculate_offsets_(true),
      first_entry_index_(-1),
      next_free_entry_(0) {}

// 생성자 Conflict
MyMemoryPlanner::MyMemoryPlanner(const int conflict_data[][3], const int conflict_data_count)
    : requirements_(nullptr),
      buffers_sorted_by_offset_(nullptr),
      buffer_sizes_sorted_(nullptr),
      buffer_ids_sorted_(nullptr),
      buffer_offsets_(nullptr),
      max_buffer_count_(0),
      buffer_count_(0),
      need_to_calculate_offsets_(true),
      first_entry_index_(-1),
      next_free_entry_(0) {
  for (int i = 0; i < conflict_data_count; ++i) {
    int input_id = conflict_data[i][0];
    int output_id = conflict_data[i][1];
    int l_value = conflict_data[i][2];
    AddConflict(input_id, output_id, l_value);
  }
}

MyMemoryPlanner::~MyMemoryPlanner() {
  // We don't own the scratch buffer, so don't deallocate anything.
}

// 메모리 플래너 초기화 함수
TfLiteStatus MyMemoryPlanner::Init(unsigned char* scratch_buffer, int scratch_buffer_size) {
  max_buffer_count_ = scratch_buffer_size / sizeof(BufferRequirements);
  requirements_ = reinterpret_cast<BufferRequirements*>(scratch_buffer);
  buffer_sizes_sorted_ = new int[max_buffer_count_];
  buffer_ids_sorted_ = new int[max_buffer_count_];
  buffer_offsets_ = new int[max_buffer_count_];
  buffers_sorted_by_offset_ = new ListEntry[max_buffer_count_];
  buffer_count_ = 0;
  return kTfLiteOk;
}

// 버퍼 추가 함수 (online 배치)
TfLiteStatus MyMemoryPlanner::AddBuffer(int size, int first_time_used, int last_time_used) {
  if (buffer_count_ >= max_buffer_count_) {
    MicroPrintf("Too many buffers (max is %d)", max_buffer_count_);
    return kTfLiteError;
  }
  BufferRequirements* current = &requirements_[buffer_count_];
  current->size = size;
  current->first_time_used = first_time_used;
  current->last_time_used = last_time_used;
  current->offline_offset = -1; // online은 -1로 설정
  ++buffer_count_;
  need_to_calculate_offsets_ = true;
  return kTfLiteOk;
}

// 버퍼 추가 함수 (offline 배치)
TfLiteStatus MyMemoryPlanner::AddBuffer(int size, int first_time_used, int last_time_used, int offline_offset) {
  if (AddBuffer(size, first_time_used, last_time_used) != kTfLiteOk) {
    return kTfLiteError;
  }
  BufferRequirements* current = &requirements_[buffer_count_ - 1];
  current->offline_offset = offline_offset;
  return kTfLiteOk;
}

// Conflict 추가 함수
void MyMemoryPlanner::AddConflict(int input_tensor_id, int output_tensor_id, int l) {
    // conflicts_ map에 conflict 정보를 추가
    conflicts_[{input_tensor_id, output_tensor_id}] = -l;

    // conflict_buffer_ids_에 input과 output 텐서 ID를 추가
    conflict_buffer_ids_.insert(input_tensor_id);
    conflict_buffer_ids_.insert(output_tensor_id);
}

// 시간적 겹침 확인 함수
bool MyMemoryPlanner::DoesEntryOverlapInTime(const ListEntry* entry, const int first_time_used, const int last_time_used) {
  const BufferRequirements& req = requirements_[entry->requirements_index];
  return !(req.last_time_used < first_time_used || req.first_time_used > last_time_used);
}

// 다음 활성 버퍼를 찾는 함수
MyMemoryPlanner::ListEntry*
MyMemoryPlanner::NextSimultaneouslyActiveBuffer(const ListEntry* start, const int first_time_used, const int last_time_used) {
  ListEntry* result = nullptr;
  ListEntry* candidate_next_entry;
  
  if (start == nullptr) {
    candidate_next_entry = &buffers_sorted_by_offset_[first_entry_index_];
  } else {
    if (start->next_entry_index == -1) {
      return nullptr;
    }
    candidate_next_entry = &buffers_sorted_by_offset_[start->next_entry_index];
  }
  
  // 순차적으로 다음 버퍼를 찾음
  do {
    if (DoesEntryOverlapInTime(candidate_next_entry, first_time_used, last_time_used)) {
      result = candidate_next_entry;
      break;
    }
    if (candidate_next_entry->next_entry_index == -1) {
      break;
    }
    candidate_next_entry = &buffers_sorted_by_offset_[candidate_next_entry->next_entry_index];
  } while (true);

  return result;
}

// 메모리 배치 함수
void MyMemoryPlanner::CalculateOffsetsIfNeeded() {
    if (!need_to_calculate_offsets_ || buffer_count_ == 0) {
        return;
    }
    need_to_calculate_offsets_ = false;

    int idx_from_head = 0;
    int idx_from_tail = buffer_count_;
    int conflict_count = 0;
    int non_conflict_count = 0;

    // 오프라인 버퍼 배치
    for (int i = 0; i < buffer_count_; ++i) {
        if (requirements_[i].offline_offset != -1) {
            buffer_sizes_sorted_[idx_from_head] = requirements_[i].size;
            buffer_ids_sorted_[idx_from_head] = i;
            buffer_offsets_[i] = requirements_[i].offline_offset;
            idx_from_head++;
        }
    }

    int idx_from_middle = idx_from_head;
    // Conflict 및 Non-Conflict 버퍼 분리 및 배치
    for (int i = 0; i < buffer_count_; ++i) {
        if (requirements_[i].offline_offset == -1) {
            if (conflict_buffer_ids_.find(i) != conflict_buffer_ids_.end()) {
                idx_from_tail--;
                buffer_sizes_sorted_[idx_from_tail] = requirements_[i].size;
                buffer_ids_sorted_[idx_from_tail] = i;
                buffer_offsets_[i] = -1;
                conflict_count++;
            } else {
                buffer_sizes_sorted_[idx_from_middle] = requirements_[i].size;
                buffer_ids_sorted_[idx_from_middle] = i;
                buffer_offsets_[i] = -1;
                idx_from_middle++;
                non_conflict_count++;
            }
        }
    }

    // non-conflict 버퍼들을 크기 순으로 정렬
    ReverseSortInPlace(&buffer_sizes_sorted_[idx_from_head],
                       &buffer_ids_sorted_[idx_from_head],
                       non_conflict_count);

    // conflict 버퍼들을 BFS로 정렬 (tail에서부터 배치)
    SortConflictTensorsBFS(&buffer_ids_sorted_[idx_from_tail], &buffer_sizes_sorted_[idx_from_tail], conflict_count);

    // 이후 버퍼 배치 로직
    first_entry_index_ = 0;
    next_free_entry_ = 1;
    ListEntry* first_entry = &buffers_sorted_by_offset_[first_entry_index_];
    first_entry->next_entry_index = -1;
    int buffer_id = buffer_ids_sorted_[0];
    first_entry->requirements_index = buffer_id;
    if (requirements_[buffer_id].offline_offset == -1) {
        buffer_offsets_[buffer_id] = 0;
    }
    first_entry->offset = buffer_offsets_[buffer_id];

    for (int i = 1; i < buffer_count_; ++i) {
        buffer_id = buffer_ids_sorted_[i];
        BufferRequirements* wanted_requirements = &requirements_[buffer_id];
        const int wanted_size = wanted_requirements->size;
        const int wanted_first_time_used = wanted_requirements->first_time_used;
        const int wanted_last_time_used = wanted_requirements->last_time_used;

        int candidate_offset = 0;
        if (wanted_requirements->offline_offset == kOnlinePlannedBuffer) {
            ListEntry* prior_entry = nullptr;
            while (true) {
                ListEntry* next_entry = NextSimultaneouslyActiveBuffer(
                    prior_entry, wanted_first_time_used, wanted_last_time_used);
                if (prior_entry) {
                    BufferRequirements* candidate_requirements =
                        &requirements_[prior_entry->requirements_index];
                    const int prior_entry_offset = prior_entry->offset + candidate_requirements->size;
                    // 현재 배치하려는 버퍼와 겹친 기존 버퍼(prior_entry)의 conflict 정보 확인
                    auto conflict_it = conflicts_.find({buffer_id, prior_entry->requirements_index});
                    if (conflict_it != conflicts_.end()) {
                        int l_value = conflict_it->second;  // conflict의 l 값
                        int conflict_offset = prior_entry->offset + l_value;  // l을 추가하여 간격 확보
                        if (conflict_offset > candidate_offset) {
                          candidate_offset = conflict_offset;
                        }
                    } else if (prior_entry_offset > candidate_offset) {
                        candidate_offset = prior_entry_offset;  // 일반적인 겹침 처리
                    }
                }
                if (next_entry == nullptr) {
                    break;
                }
                const int gap = next_entry->offset - candidate_offset;
                if (gap >= wanted_size) {
                    break;
                }
                prior_entry = next_entry;
            }
        } else {
            candidate_offset = wanted_requirements->offline_offset;
        }
        buffer_offsets_[buffer_id] = candidate_offset;
        ListEntry* new_entry = &buffers_sorted_by_offset_[next_free_entry_];
        new_entry->offset = candidate_offset;
        new_entry->requirements_index = buffer_id;
        const int new_entry_index = next_free_entry_;
        ++next_free_entry_;

        if (first_entry->offset > candidate_offset) {
            first_entry = new_entry;
            first_entry->next_entry_index = first_entry_index_;
            first_entry_index_ = new_entry_index;
        } else {
            ListEntry* current_entry = first_entry;
            while (true) {
                const int next_entry_index = current_entry->next_entry_index;
                if (next_entry_index == -1) {
                    current_entry->next_entry_index = new_entry_index;
                    new_entry->next_entry_index = -1;
                    break;
                }
                ListEntry* next_entry = &buffers_sorted_by_offset_[next_entry_index];
                if (next_entry->offset > candidate_offset) {
                    new_entry->next_entry_index = current_entry->next_entry_index;
                    current_entry->next_entry_index = new_entry_index;
                    break;
                }
                current_entry = next_entry;
            }
        }
    }
}

// 최대 메모리 크기 반환
size_t MyMemoryPlanner::GetMaximumMemorySize() {
  CalculateOffsetsIfNeeded();
  size_t max_size = 0;
  for (int i = 0; i < buffer_count_; ++i) {
    max_size = std::max(max_size, static_cast<size_t>(buffers_sorted_by_offset_[i].offset + requirements_[buffers_sorted_by_offset_[i].requirements_index].size));
  }
  return max_size;
}

// 메모리 플랜 출력 함수
void MyMemoryPlanner::PrintMemoryPlan() {
  CalculateOffsetsIfNeeded();  // 오프셋이 필요한지 확인하고 계산
  // 각 버퍼의 기본 정보를 출력
  for (int i = 0; i < buffer_count_; ++i) {
    MicroPrintf("%c (id=%d): size=%d, offset=%d, first_used=%d last_used=%d",
                GetOrdinalCharacter(i), i, requirements_[i].size,
                buffer_offsets_[i], requirements_[i].first_time_used,
                requirements_[i].last_time_used);
  }
  MicroPrintf("\n==========================================================================================\n");

  // 각 버퍼를 정렬한 형태로, 추가하는 순서대로 출력
  for (int i = 0; i < buffer_count_; ++i) {
    MicroPrintf("%c (id=%d): size=%d, offset=%d, first_used=%d last_used=%d",
                GetOrdinalCharacter(buffer_ids_sorted_[i]), buffer_ids_sorted_[i], requirements_[buffer_ids_sorted_[i]].size,
                buffer_offsets_[buffer_ids_sorted_[i]], requirements_[buffer_ids_sorted_[i]].first_time_used,
                requirements_[buffer_ids_sorted_[i]].last_time_used);
  }
  MicroPrintf("\n==========================================================================================\n");

  // 메모리 레이아웃을 시각적으로 표현
  constexpr int kLineWidth = 120;
  int max_size = kLineWidth;
  int max_time = 0;

  // 최대 메모리 크기 및 최대 사용 시간 계산
  for (int i = 0; i < buffer_count_; ++i) {
    BufferRequirements* requirements = &requirements_[i];
    const int offset = buffer_offsets_[i];
    const int last_time_used = requirements->last_time_used;
    const int size = offset + requirements->size;
    if (size > max_size) {
        max_size = size;
    }
    if (last_time_used > max_time) {
        max_time = last_time_used;
    }
  }

  // 버퍼들이 차지하는 공간을 표현할 문자열 라인 초기화
  char line[kLineWidth + 1];
  
  // 시간에 따른 메모리 사용 상태를 출력
  for (int t = 0; t <= max_time; ++t) {
    // 라인을 초기화
    for (int c = 0; c < kLineWidth; ++c) {
      line[c] = '.';
    }

    int memory_use = 0;

    // 각 버퍼의 메모리 사용 상태를 확인
    for (int i = 0; i < buffer_count_; ++i) {
      BufferRequirements* requirements = &requirements_[i];
      
      // 현재 시간 t에 사용되지 않는 버퍼는 제외
      if (t < requirements->first_time_used || t > requirements->last_time_used) {
        continue;
      }

      const int offset = buffer_offsets_[i];
      if (offset == -1) {
        continue; // 유효하지 않은 오프셋 무시
      }

      const int size = requirements->size;
      const int memory_end = offset + size; // offset + size 계산
      memory_use = std::max(memory_use, memory_end); // 가장 큰 offset + size 추적

      // 해당 버퍼의 메모리 영역을 라인에 표시
      const int line_start = (offset * kLineWidth) / max_size;
      const int line_end = ((offset + size) * kLineWidth) / max_size;
      for (int n = line_start; n < line_end; ++n) {
        if (line[n] == '.') {
          line[n] = GetOrdinalCharacter(i);
        } else {
          line[n] = '!'; // 기존에 텐서가 배치되어 있는 경우 (Conflict 출력)
        }
      }
    }
    // 라인의 끝에 널 문자를 추가하여 출력
    line[kLineWidth] = '\0';

    // 시간 t에 해당하는 메모리 상태 출력
    MicroPrintf("%s%d: %s (%dk)", t < 10 ? " " : "", t, (const char*)line,
                (memory_use + 1023) / 1024);
  }

  MicroPrintf("\n==========================================================================================\n");
  MicroPrintf("Maximum Memory Size: %d", max_size);
}


int MyMemoryPlanner::GetBufferCount() { return buffer_count_; }

// 특정 버퍼 오프셋 반환 함수
TfLiteStatus MyMemoryPlanner::GetOffsetForBuffer(int buffer_index, int* offset) {
  CalculateOffsetsIfNeeded();
  if (buffer_index < 0 || buffer_index >= buffer_count_) {
    return kTfLiteError;
  }
  *offset = buffers_sorted_by_offset_[buffer_index].offset;
  return kTfLiteOk;
}

}   // namespace tflite
