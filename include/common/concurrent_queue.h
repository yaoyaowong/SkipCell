#ifndef COMMON_CONCURRENT_QUEUE
#define COMMON_CONCURRENT_QUEUE

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace powerlaw_ann {

template <typename T>
class concurrent_queue_t {
  typedef std::chrono::microseconds chrono_us_t;
  typedef std::unique_lock<std::mutex> mutex_locker_t;

  std::queue<T> q;
  std::mutex mut;
  std::mutex push_mut;
  std::mutex pop_mut;
  std::condition_variable push_cv;
  std::condition_variable pop_cv;
  T null_T;

public:
  concurrent_queue_t() {}

  concurrent_queue_t(T null_t) { this->null_T = null_t; }

  ~concurrent_queue_t() {
    this->push_cv.notify_all();
    this->pop_cv.notify_all();
  }

  // queue stats
  uint64_t size() {
    mutex_locker_t lk(this->mut);
    uint64_t ret = q.size();
    lk.unlock();
    return ret;
  }

  bool empty() { return (this->size() == 0); }

  // PUSH BACK
  void push(T& new_val) {
    mutex_locker_t lk(this->mut);
    this->q.push(new_val);
    lk.unlock();
  }

  template <class Iterator>
  void insert(Iterator iter_begin, Iterator iter_end) {
    mutex_locker_t lk(this->mut);
    for (Iterator it = iter_begin; it != iter_end; it++) {
      this->q.push(*it);
    }
    lk.unlock();
  }

  // POP FRONT
  T pop() {
    mutex_locker_t lk(this->mut);
    if (this->q.empty()) {
      lk.unlock();
      return this->null_T;
    } else {
      T ret = this->q.front();
      this->q.pop();
      // powerlaw_ann::cout << "thread_id: " << std::this_thread::get_id() <<
      // ", ctx: "
      // << ret.ctx << "\n";
      lk.unlock();
      return ret;
    }
  }

  // register for notifications
  void wait_for_push_notify(chrono_us_t wait_time = chrono_us_t{10}) {
    mutex_locker_t lk(this->push_mut);
    this->push_cv.wait_for(lk, wait_time);
    lk.unlock();
  }

  void wait_for_pop_notify(chrono_us_t wait_time = chrono_us_t{10}) {
    mutex_locker_t lk(this->pop_mut);
    this->pop_cv.wait_for(lk, wait_time);
    lk.unlock();
  }

  // just notify functions
  void push_notify_one() { this->push_cv.notify_one(); }
  void push_notify_all() { this->push_cv.notify_all(); }
  void pop_notify_one() { this->pop_cv.notify_one(); }
  void pop_notify_all() { this->pop_cv.notify_all(); }
};
} // namespace powerlaw_ann

#endif // COMMON_CONCURRENT_QUEUE
