from api.Api import Api, Method


class AgentApi(Api):

    @property
    def controller_prefix(self) -> str:
        return "agent"

    def set_position(self, x, y, z):
        return self.call("position", Method.POST, data={"X": x, "Y": y, "Z": z})

    def get_position(self):
        return self.call("position", Method.GET)

    def set_transform(self, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
        data = {
            "Position": {
                "X": pos_x,
                "Y": pos_y,
                "Z": pos_z
            },
            "Rotation": {
                "X": rot_x,
                "Y": rot_y,
                "Z": rot_z
            }
        }

        return self.call("transform", Method.POST, data=data)

    def get_transform(self):
        return self.call("transform", Method.GET)