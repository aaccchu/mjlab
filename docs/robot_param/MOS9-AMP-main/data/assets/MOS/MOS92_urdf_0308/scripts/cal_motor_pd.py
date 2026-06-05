import math
from typing import Dict

ENCOS_A4310 = {
  "reduction_ratio": 36,
  "peak_speed_rating_rpm": 89,
  "peak_torque_rating_Nm": 36,
  "rotor_inertia_kgmm2": 18.2,
}

ENCOS_A6408 = {
  "reduction_ratio": 25,
  "peak_speed_rating_rpm": 149,
  "peak_torque_rating_Nm": 60,
  "rotor_inertia_kgmm2": 62.254,
}

FREQUENCE = 8.0  # Hz
OMEGA_N = 2.0 * math.pi * FREQUENCE  # rad/s
ZETA = 2.0


def kgmm2_to_kgm2(value_kgmm2: float) -> float:
  return value_kgmm2 * 1e-6


def compute_motor_params() -> Dict[str, Dict[str, float]]:
  motors = {
    "ENCOS_A4310": ENCOS_A4310,
    "ENCOS_A6408": ENCOS_A6408,
  }

  results: Dict[str, Dict[str, float]] = {}
  for name, motor in motors.items():
    reduction_ratio = float(motor["reduction_ratio"])
    tau_peak = float(motor["peak_torque_rating_Nm"])
    rotor_inertia_kgmm2 = float(motor["rotor_inertia_kgmm2"])
    rpm = float(motor["peak_speed_rating_rpm"])
    rad_per_sec = rpm * 2.0 * math.pi / 60.0

    rotor_inertia_kgm2 = kgmm2_to_kgm2(rotor_inertia_kgmm2)

    # 输出端转动惯量：I_out = I_rotor * N^2
    I_out = rotor_inertia_kgm2 * reduction_ratio**2

    # 二阶系统参数
    k_p = I_out * OMEGA_N**2
    k_d = 2.0 * ZETA * math.sqrt(k_p * I_out)
    alpha = 0.25 * tau_peak / k_p

    results[name] = {
      "reduction_ratio": reduction_ratio,
      "tau_peak": tau_peak,
      "speed_rad_per_sec": rad_per_sec,
      "rotor_inertia_kgmm2": rotor_inertia_kgmm2,
      "rotor_inertia_kgm2": rotor_inertia_kgm2,
      "I_out": I_out,
      "frequence": FREQUENCE,
      "omega_n": OMEGA_N,
      "zeta": ZETA,
      "kp": k_p,
      "kd": k_d,
      "alpha": alpha,
    }

  return results


def main() -> None:
  results = compute_motor_params()
  for name, vals in results.items():
    print(f"\n{name}")
    print(f"  reduction_ratio              : {vals['reduction_ratio']:.6f}")
    print(f"  tau_peak (Nm)                : {vals['tau_peak']:.6f}")
    print(f"  speed_peak (rad/s)           : {vals['speed_rad_per_sec']:.6f}")
    print(f"  rotor_I (kg*mm^2)            : {vals['rotor_inertia_kgmm2']:.6f}")
    print(f"  rotor_I (kg*m^2)             : {vals['rotor_inertia_kgm2']:.9e}")
    print(f"  I_out (kg*m^2)               : {vals['I_out']}")
    print(f"  natural frequence (Hz)       : {vals['frequence']}")
    print(f"  omega_n (rad/s)              : {vals['omega_n']}")
    print(f"  zeta                         : {vals['zeta']}")
    print(f"  kp                           : {vals['kp']}")
    print(f"  kd                           : {vals['kd']}")
    print(f"  alpha                        : {vals['alpha']}")


if __name__ == "__main__":
  main()


"""
ENCOS_A4310
  reduction_ratio              : 36.000000
  tau_peak (Nm)                : 36.000000
  speed_peak (rad/s)           : 9.320058
  rotor_I (kg*mm^2)            : 18.200000
  rotor_I (kg*m^2)             : 1.820000000e-05
  I_out (kg*m^2)               : 0.0235872
  natural frequence (Hz)       : 8.0
  omega_n (rad/s)              : 50.26548245743669
  zeta                         : 2.0
  kp                           : 59.595861229919976
  kd                           : 4.742487951280203
  alpha                        : 0.15101719841379804

ENCOS_A6408
  reduction_ratio              : 25.000000
  tau_peak (Nm)                : 60.000000
  speed_peak (rad/s)           : 15.603244
  rotor_I (kg*mm^2)            : 62.254000
  rotor_I (kg*m^2)             : 6.225400000e-05
  I_out (kg*m^2)               : 0.03890874999999999
  natural frequence (Hz)       : 8.0
  omega_n (rad/s)              : 50.26548245743669
  zeta                         : 2.0
  kp                           : 98.30757638166668
  kd                           : 7.823068362263157
  alpha                        : 0.1525823395519833
"""
