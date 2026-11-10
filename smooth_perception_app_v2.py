import desktop_app_v2

from smooth_perception_renderer_v2 import draw_smooth_perception


class SmoothPerceptionApp(desktop_app_v2.DesktopSimulationApp):
    def redraw(self) -> None:
        original_draw_board = desktop_app_v2.draw_board
        desktop_app_v2.draw_board = draw_smooth_perception
        try:
            super().redraw()
        finally:
            desktop_app_v2.draw_board = original_draw_board


if __name__ == "__main__":
    SmoothPerceptionApp().show()
