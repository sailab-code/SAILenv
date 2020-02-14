from api.Api import Api, Method


class CameraApi(Api):
    @property
    def controller_prefix(self) -> str:
        return "camera"

    def get_main_frame(self):
        return self.call("main", Method.GET)

    def get_category_frame(self):
        return self.call("category", Method.GET)

    def get_object_frame(self):
        return self.call("object", Method.GET)

    def __init__(self, object_frame_active: bool = True, category_frame_active: bool = True, flow_frame_active: bool = True):
        super().__init__()
        self.object_frame_active : bool = object_frame_active
        self.category_frame_active: bool = category_frame_active
        self.flow_frame_active: bool = flow_frame_active

    def get_frame(self):
        params = {
            "Main": True,
            "Category": self.category_frame_active,
            "Object": self.object_frame_active,
            "Optical": self.flow_frame_active
        }
        return self.call("frame", Method.GET, data=params)