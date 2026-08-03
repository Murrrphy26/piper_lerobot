# tools/test

只读排查脚本（不改臂参数）。

## `query_master_impedance_params.py`

查询主臂示教相关手感参数：控制模式、MIT/安装位、示教行程与 `teaching_friction`、固件版本等。

```bash
conda activate piper
cd /home/agilex/code/yjw/piper
bash scripts/bringup_can.sh   # 需要本机 sudo 密码
python tools/test/query_master_impedance_params.py
python tools/test/query_master_impedance_params.py --can can0
```

### 常见误用

```python
import piper_sdk as piper
piper.ArmMsgParamEnquiryAndConfig(4)  # ❌ 只构造消息，不连臂、不查询
```

正确：

```python
from piper_sdk import C_PiperInterface_V2
arm = C_PiperInterface_V2("can0")
arm.ConnectPort(piper_init=False)
arm.ArmParamEnquiryAndConfig(4)  # 发出查询
print(arm.GetGripperTeachingPendantParamFeedback())
```
