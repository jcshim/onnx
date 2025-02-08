안녕 JC! 이번에는 아주 간단하게 두 개의 스칼라 값을 더하는 ONNX 모델을 만들어볼게. 이 예제는 ONNX 라이브러리로 모델을 만들고, onnxruntime으로 실행하는 예제야. 실행하면 2.0과 3.0을 더해서 5.0이 나오는 걸 확인할 수 있어.

아래 코드를 그대로 복사해서 실행해봐. (참고로, 이 코드를 실행하려면 `onnx`와 `onnxruntime` 패키지가 설치되어 있어야 해. 설치는 `pip install onnx onnxruntime` 명령어로 할 수 있어.)

코드오류

### 코드 설명
1. **모델 생성**  
   - `onnx.helper.make_node`로 `Add` 노드를 만들어 두 입력("A", "B")의 합을 출력("C")하도록 했어.
   - `onnx.helper.make_tensor_value_info`를 이용해 입력과 출력의 텐서 정보를 정의했어. 여기서는 스칼라 값이니까 `shape`를 `[]`로 했어.
   - 이 노드들을 그래프로 묶은 후, 모델로 만들고 `add.onnx` 파일로 저장했어.

2. **모델 실행**  
   - `onnxruntime`의 `InferenceSession`을 사용해 저장한 ONNX 모델을 로딩했어.
   - NumPy 배열로 스칼라 값 `2.0`과 `3.0`을 입력한 후, `session.run`을 통해 추론을 수행했어.
   - 결과는 `[5.]`가 출력될 거야.

----------
안녕 JC! 오류 메시지를 보니, ONNX Runtime에서 지원하는 opset 버전이 21까지인데 모델이 opset 22로 생성돼서 발생한 문제야.  
즉, 모델 생성 시에 명시적으로 opset을 21로 설정해주면 해결돼.

아래에 수정된 코드를 줄게:

```python
import onnx
import onnxruntime as ort
import numpy as np

# 1. ONNX 모델 생성
# 두 스칼라 입력 A와 B를 받아서, Add 노드로 더한 후 결과 C를 출력하는 간단한 그래프

# Add 노드 생성
node = onnx.helper.make_node(
    op_type="Add",
    inputs=["A", "B"],
    outputs=["C"]
)

# 입력 텐서 정보 (스칼라니까 shape은 [])
A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [])
B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [])
# 출력 텐서 정보
C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [])

# 그래프 생성
graph = onnx.helper.make_graph(
    nodes=[node],
    name="add_graph",
    inputs=[A, B],
    outputs=[C]
)

# opset 버전을 명시적으로 21로 지정 (기본 도메인 "")
opset = onnx.helper.make_opsetid("", 21)
model = onnx.helper.make_model(graph, producer_name="onnx-example", opset_imports=[opset])

# 모델 저장
onnx.save(model, "add.onnx")
print("ONNX 모델이 'add.onnx'로 저장되었어.")

# 2. onnxruntime으로 모델 실행 (추론)
session = ort.InferenceSession("add.onnx")

# 입력 데이터: 스칼라 값 2.0과 3.0
a_val = np.array(2.0, dtype=np.float32)
b_val = np.array(3.0, dtype=np.float32)

# 추론 실행
result = session.run(None, {"A": a_val, "B": b_val})
print("추론 결과:", result[0])
```

### 코드 설명
1. **opset 버전 설정**  
   `opset = onnx.helper.make_opsetid("", 21)` 코드를 추가해서 기본 도메인(`""`)의 opset 버전을 21로 명시했어. 이렇게 하면 ONNX Runtime이 모델을 올바르게 인식할 수 있어.

2. **모델 생성 및 저장**  
   수정된 `opset_imports`를 이용해 모델을 생성하고, `"add.onnx"`로 저장했어.

3. **모델 실행**  
   onnxruntime의 `InferenceSession`으로 모델을 불러와서, 입력값 2.0과 3.0을 더한 결과(5.0)를 확인할 수 있어.

이제 위 코드를 실행하면 오류 없이 추론 결과를 확인할 수 있을 거야. 더 궁금한 점이나 다른 문제가 있으면 언제든 말해줘, JC!
