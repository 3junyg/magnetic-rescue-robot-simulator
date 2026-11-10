from pathlib import Path
import subprocess
import sys
import tkinter as tk


def main() -> None:
    folder = Path(__file__).resolve().parent
    window = tk.Tk()
    window.title('V2 로컬 시뮬레이션 앱')
    window.geometry('370x330')
    tk.Label(window, text='실행할 앱을 선택하세요', font=('Malgun Gothic', 13)).pack(pady=14)
    entries = (
        ('기본 시뮬레이션', 'basic_app_v2.py'),
        ('Smooth 인식 화면', 'smooth_app_v2.py'),
        ('추가 노이즈 시뮬레이션', 'noise_app_v2.py'),
        ('노이즈 대응 모델 시뮬레이션', 'noise_robust_app_v2.py'),
        ('좁은 통로 시각화 시험', 'narrow_navigation_app_v2.py'),
    )
    for label, filename in entries:
        tk.Button(window, text=label, font=('Malgun Gothic', 11), width=30,
                  command=lambda name=filename: subprocess.Popen([sys.executable, str(folder / name)], cwd=folder)).pack(pady=5)
    window.mainloop()


if __name__ == '__main__':
    main()
