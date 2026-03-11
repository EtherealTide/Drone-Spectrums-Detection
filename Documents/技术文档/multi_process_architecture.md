# 基于多进程的系统架构指南 (Multi-process System Architecture Guide)

本文档基于本项目的实际代码（`main.py`、`ipc.py`、`data_process.py`、`algorithms.py`、`communication.py`、`state.py`等）详细解析了多进程系统的设计思想、开发流程、业务流转以及各类核心技术与函数的正确用法。

---

## 1. 系统架构概述 (Architecture Overview)

本项目整体采用“一主多从”的多进程架构（基于Python标准库 `multiprocessing`），进程间以**共享内存(Shared Memory)**和**队列(Queue)**实现通信。主要划分如下：

- **主进程 (Main Process / `main.py`)**: 运行 PyQt6 UI、管理全局状态 (`State`)、分配全局 IPC（进程间通信）资源，并通过定时器轮询拉取子进程的状态信息 (FPS、检测结果、连接状态等)。
- **进程 1 - 通信进程 (Communication Process / `communication.py`)**: 负责通过 TCP Socket 连接硬件，接收并解析原始 FFT 数据包，并通过 `Queue` 将帧数据发送给数据处理进程。
- **进程 2 - 数据处理进程 (DataProcessor / `data_process.py`)**: 核心业务进程。从通信进程读取 FFT 数据并应用滤波和组装逻辑，之后将生成的“瀑布图”和“频谱数据”**写入共享内存**。
- **进程 3 - 算法检测进程 (DroneDetector / `algorithms.py`)**: 从**共享内存**中读取“瀑布图”像素数据，使用 YOLO (PyTorch/OpenVINO) 进行目标检测，并将绘制了边界框的结果画面再写入共享内存供主进程 UI 显示。

---

## 2. 核心执行流程 (Execution Flow)

### 2.1 主进程与子进程的初始化流程
整个系统的启动流程由 `main.py` 中的 `DroneDetectionSystem` 控制：

1. **环境预设**: 调用 `mp.set_start_method("spawn", force=True)`，确保跨平台（特别是 Windows）的子进程干净启动。
2. **应用实例**: 创建 Qt 应用实例 `QApplication(sys.argv)`。
3. **分配资源**: 主进程通过 `ipc.py` 创建全局的 `SharedMemory`（瀑布图、频谱、检测图），并创建所有全局的队列 (`Queue`)、锁 (`Lock`) 和共享值 (`Value`)。
4. **启动子进程**: 主进程使用 `mp.Process` 依次启动三个子进程（检测进程、数据进程、通信进程）。
   - *注意启动顺序与预热*：检测进程和数据进程最先启动，因为目标检测模型（如 YOLO）需要时间加载和预热。
   - 所有子进程的 `daemon=True`，保证主进程崩溃时子进程自动销毁。
5. **UI 初始化**: 启动 `Window` 主界面，并将所有 UI 控件的信号槽连接到控制逻辑。
6. **启动轮询**: 启动 `QTimer` (如 40ms/25Hz)，主线程开始轮询各个子进程的统计信息队列 (`xxx_stats_q`)，并触发 UI 更新。
7. **进入主循环**: 运行 `app.exec()`，主进程让出控制权给 Qt 事件循环。

### 2.2 子进程生命周期 (以数据处理进程为例)
子进程被 `mp.Process` 启动后，会执行其指定的 `target` 函数。

1. **入口函数 (`data_processor_process`)**: 
   - 独立配置 `logging`（日志处理器不跨进程继承）。
   - 实例化具体的业务处理类（如 `DataProcessor`）。
   - **空闲循环**: 进入 `while system_running.value`，监听控制队列 (`dp_ctrl_q`)。
2. **待机与配置**: 处于待机状态时，若收到 `SET_PARAM`，则仅更新内部属性；若收到 `START`，则调用业务类的 `run()`。
3. **正式运行 (`run` 方法)**:
   - 启动内部线程（如数据接收线程、图像转换线程），执行计算密集型操作。
   - `run()` 成为一个阻塞循环，不断处理数据，同时偶尔检查控制指令。
   - 收到 `STOP` 时，修改内部状态标志 `_running = False`，退出阻塞循环返回上一层。
4. **资源清理**: 当主进程退出（`system_running.value == False`），子进程跳出所有循环。释放 numpy 视图映射（`del`, `gc`），安全关闭自身的 `SharedMemory` 句柄，进程安全退出。

---

## 3. 业务流转：UI 界面修改参数如何生效？

系统采用单向数据流进行参数同步，避免了复杂的双向状态一致性问题。

### UI 修改后端参数的完整链路：
1. **用户操作 UI**: 用户在界面（如配置面板）修改了某个参数，例如修改 `FFT_Length` 或修改了某种滤波器的开关。
2. **UI 触发信号**: UI 组件（例如 `ConfigInterface`）通过 Qt 信号发出改变的请求：`parameter_change_request.emit(group, name, value)`。
3. **系统处理 (`main.py` 中的槽函数)**:
   - `_handle_parameter_change` 接收到了信号。
   - 首先同步到全局状态管理器 (`self.state.set_parameter`)，这会将参数写回本地 JSON，并在主进程维护最新状态。
   - 封装为控制指令：`cmd = {"cmd": "SET_PARAM", "group": group, "name": name, "value": value}`。
4. **分发到队列**: 根据参数的作用域，将该指令推入对应子进程的控制队列（如 `dp_ctrl_q.put_nowait(cmd)`）。
5. **子进程响应**:
   - 子进程在它的大循环或内部线程中，定期使用 `.get_nowait()` 检查其控制队列。
   - 读取到 `SET_PARAM` 后，调用内部的方法（如 `_handle_command`），对自身实例的变量进行覆盖。为防止多线程并发问题，更新核心属性时通常加锁（如使用 `with self.data_lock:`）。如果是硬件参数，通信进程还会将指令封包通过 Socket 发送到设备。

---

## 4. 共享内存的设计与用法 (Shared Memory)

### 澄清一个常见误区：共享内存不是队列！
在 Python 多进程中，**共享内存 (`SharedMemory`)** 和 **队列 (`mp.Queue`)** 是两种完全不同的 IPC（进程间通信）机制，用途截然不同：

- **队列 (`mp.Queue`)**：本质上是一个先进先出 (FIFO) 的管道。当你要把一个命令（如 `START`, `SET_PARAM`）或状态字典（如 FPS）发给另一个进程时，你把它放进队列。接收方从队列里把它拿出来。拿出来之后，数据就从队列里消失了。数据在传递过程中会被**序列化 (Pickle)**和**拷贝**。
- **共享内存 (`SharedMemory`)**：本质上是一块**所有进程都能同时看到、直接读写的物理内存区域**。在本项目中，我们把它当成一个巨大的白板。写进程（DataProcessor）在上面画瀑布图，读进程（Detector 和 UI）直接看这块白板的内容。**它没有“先进先出”的概念，新数据直接覆盖老数据。它不需要序列化，实现了零拷贝 (Zero-copy) 传输。**

### 为什么用共享内存而不用队列传图像？
如果使用 `multiprocessing.Queue` 传递大量数据（如一帧 60MB 的瀑布图画面），Python 需要将这 60MB 数据序列化再压入队列、接收端再反序列化，这个过程会产生极大的 CPU 负担并引发显著的通信延迟，导致帧率断崖式下跌。
对于图像、巨大的一维数组（FFT、像素等），**共享内存 (Shared Memory)** 配合 Numpy 视图 (`ndarray view`) 是原生且高性能的方案，直接在物理内存上划出一块由所有进程共同读写的区域。

### 怎么创建共享内存？ (建一次，常驻内存)
通常在 `ipc.py` 这种专用模块中由主进程进行分配。**建议直接按最大尺寸预分配 (`size` 参数)**，以应对运行时动态调整（例如 FFT 长度变化）的情况。
```python
from multiprocessing.shared_memory import SharedMemory
import numpy as np

# 1. 估算字节需求量并创建（不是建队列，是建一块规定大小的连续内存）
SHM_SHAPE = (1024, 20)
SHM_DTYPE = np.float32
n_bytes = int(np.prod(SHM_SHAPE)) * np.dtype(SHM_DTYPE).itemsize

# 2. 从主进程创建并初始化为0
shm_block = SharedMemory(create=True, size=n_bytes)
np.frombuffer(shm_block.buf, dtype=SHM_DTYPE)[:] = 0

# 主进程只需保存 name 和对象，供日后别的进程连接和最终回收
name = shm_block.name
```

### 怎么使用共享内存？ (用 Numpy 视图读取，需考虑并发)
子进程通过被传入的 `name` 字符串，通过操作系统找到并连接到这块已经建好的物理内存。然后将其映射为 Numpy 数组结构 (`ndarray view`)。
此后，这块内存就像一个普通的、大家都能访问的全局 numpy 数组：
```python
# 1. 连接现有共享内存空间
self._shm = SharedMemory(name=shm_name)

# 2. 将这段连续内存“塑形”为多维 Numpy 数组视图，实现零拷贝读写
self._array_view = np.frombuffer(
    self._shm.buf, dtype=np.uint8
).reshape(EXPECTED_SHAPE)

# 3. 往共享内存写数据时，建议使用 Lock 避免读写冲突
# 因为这不是队列，没有天然的顺序保护。如果不加锁，可能读进程读到一半，写进程刚好覆盖了另一半，导致图像撕裂。
with self.data_lock:
    self._array_view[:512, :20] = new_data
```

---

## 5. 多进程相关核心函数与同步原语大全

### 5.1 启动类函数
- `mp.set_start_method("spawn", force=True)`
  - **作用**: 强制所有子进程启动环境等同于 `spawn`。Windows 默认即此，Linux 默认是 `fork`。为了全平台一致性和防止一些与 Qt 的奇怪崩溃，必须在 main 最开头设定。
- `mp.Process(target, args, daemon=True, name)`
  - **作用**: 创建子进程。
  - **关键参数**: 
    - `target`: 子进程将执行的函数。
    - `args`: 一个元组，传入包含队列、锁、内存 name 的组合。
    - `daemon=True`: 守护进程属性。主进程一旦终结，即使子进程还在执行死循环，操作系统也会将子进程强制杀死。

### 5.2 队列操作 (`mp.Queue`)
**适用场景**: 进程间轻量级通信（命令、配置字典、统计数据）。
- `q.put_nowait(item)`
  - **作用**: 将数据存入队列，如果队列满了则立即抛出 `queue.Full` 异常而不是死等。
  - **使用范例 (防积压更新法)**: 在发送帧率、FPS等无关历史的状态时，满了就丢掉旧的。
    ```python
    try:
        q.put_nowait(stats)
    except queue.Full:
        try:
            q.get_nowait() # 丢弃最老的一条
            q.put_nowait(stats)
        except Exception:
            pass
    ```
- `q.get(timeout=1.0)`
  - **作用**: 阻塞获取队列内容，最多等 1 秒。
  - **使用场景**: 让处于待机闲置状态的子进程休眠（让出 CPU 时间片），若 1 秒没收到就进入下一个循环继续判断全局终止开关。
- `q.get_nowait()`
  - **作用**: 立即获取队列顶部元素，如果队列为空抛出 `queue.Empty`。
  - **使用场景**: 在密集的图片处理或检测循环中，不希望浪费任何时间被阻塞，只是快速“瞥一眼”有没有最新的 UI 控制命令。

### 5.3 锁操作 (`mp.Lock`)
**适用场景**: 保证跨进程数据的读写互斥（比如共享内存画面刷新），防止“撕裂”（读到了写了一半的帧）。
- `lock = mp.Lock()`: 在主进程创建。
- `with lock:`
  - **作用**: 上下文内锁定资源。由于上锁会有不小的开销并将多个进程变为串行运行，上锁的代码块应该**越短越好**，仅包裹最后的内存拷贝：
    ```python
    # BAD: 在锁里面算很慢的操作
    with memory_lock:
         data = cv.resize(img)
         shared_mem[:] = data
         
    # GOOD: 算完再锁
    data = cv.resize(img) 
    with memory_lock:
         shared_mem[:] = data
    ```

### 5.4 共享值操作 (`mp.Value`)
**适用场景**: 全局的小型数字开关（布尔值、计数器）。
- `val = mp.Value("b", True)` / `mp.Value("i", 0)`
  - **作用**: 创建全局类型指针，`'b'` 为布尔值，`'i'` 为整数。
- `val.value`
  - **作用**: 获取或设置实际值。如主循环 `while system_running.value:`，主进程只要将其置为 `False`，所有子进程就会在下一次循环退出。

### 5.5 共享内存安全操作 (`SharedMemory`)
- `shm = SharedMemory(create=True, size=...)` (主进程创建)
- `shm = SharedMemory(name="...")` (子进程连接)
- `shm.close()`
  - **作用**: 通知系统当前进程不再使用这段共享映射区。每个连接过它的进程结束前都必须执行。
- `shm.unlink()`
  - **作用**: 请求系统彻底抹除这段内存块。**只有主进程可以且只能执行一次！**在所有子进程确认死掉之后在主进程调用。
- **防止崩溃的关键清理操作**: 如果使用了 `np.frombuffer` 创建映射，必须先使用 `del view_array` 删除视图引用。否则这部分内存还有残余关联，底层直接强行 `close` 往往产生非法访问错误 (Access Violation)。
