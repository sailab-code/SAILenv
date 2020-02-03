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
