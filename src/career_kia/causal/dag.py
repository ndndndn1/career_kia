"""제조 공정 도메인 DAG.

자동차 파워트레인 가공 라인의 공정 파라미터 - 베어링 상태 - 기계 고장
사이의 인과 관계에 대한 **도메인 지식 기반 초기 가정**을 인코딩한다.

실데이터에 PC/GES 같은 구조 학습을 적용해 교차 검증은 할 수 있지만,
최종 DAG 는 현장 엔지니어의 리뷰를 받아 확정해야 한다는 것이 본 모듈의
입장이다 — "가정을 명시적으로 적는 것" 이 인과추론의 전제.

변수::

    Type(품질등급, L/M/H)   ── exogenous
    Tool wear (공구 마모)    ── exogenous (시간 누적)
    Air temperature         ── exogenous (환경)
    Rotational speed        ── exogenous (오퍼레이터 제어)
    Torque                  ── Tool wear, Rotational speed 의 영향 (confounder 관점)
    Process temperature     ── Air temperature, Torque 의 영향
    Vibration RMS           ── Tool wear, Rotational speed, Torque 의 영향
    Machine failure         ── Process temp, Tool wear, Torque, Vibration, Type
"""

from __future__ import annotations

# GML 형식 DAG 문자열 — DoWhy CausalModel 에 그대로 전달 가능
CAUSAL_GRAPH_GML = """
graph [
  directed 1
  node [id 0 label "Type"]
  node [id 1 label "Tool_wear"]
  node [id 2 label "Air_temperature"]
  node [id 3 label "Rotational_speed"]
  node [id 4 label "Torque"]
  node [id 5 label "Process_temperature"]
  node [id 6 label "Vibration_RMS"]
  node [id 7 label "Machine_failure"]

  edge [source 1 target 4]
  edge [source 3 target 4]

  edge [source 2 target 5]
  edge [source 4 target 5]

  edge [source 1 target 6]
  edge [source 3 target 6]
  edge [source 4 target 6]

  edge [source 0 target 7]
  edge [source 1 target 7]
  edge [source 4 target 7]
  edge [source 5 target 7]
  edge [source 6 target 7]
]
"""

# 주요 변수명 맵 (실컬럼 ↔ DAG 노드명)
NODE_TO_COLUMN: dict[str, str] = {
    "Type": "Type",
    "Tool_wear": "Tool wear [min]",
    "Air_temperature": "Air temperature [K]",
    "Rotational_speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Process_temperature": "Process temperature [K]",
    "Vibration_RMS": "t_rms_mean",
    "Machine_failure": "Machine failure",
}


def get_default_graph() -> str:
    return CAUSAL_GRAPH_GML.strip()
