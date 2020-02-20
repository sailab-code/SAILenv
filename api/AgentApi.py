from api.Api import Api, Method


class AgentApi(Api):

    @property
    def controller_prefix(self) -> str:
        return "agent"

    def set_position(self, agent_id: int, x, y, z):
        return self.call(f"{agent_id}/position", Method.POST, data={"X": x, "Y": y, "Z": z})

    def get_position(self, agent_id: int):
        return self.call(f"{agent_id}/position", Method.GET)

    def set_transform(self, agent_id: int, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
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

        return self.call(f"{agent_id}/transform", Method.POST, data=data)

    def get_transform(self, agent_id: int):
        return self.call(f"{agent_id}/transform", Method.GET)

    def register(self):
        return self.call("register", Method.POST)

    def delete(self, agent_id: int):
        return self.call(f"{agent_id}/delete", Method.POST)

    def get_frame(self,
                  agent_id: int,
                  main_frame_active: bool = True,
                  object_frame_active: bool = True,
                  category_frame_active: bool = True,
                  flow_frame_active: bool = True
    ):
        params = {
            "Main": main_frame_active,
            "Category": category_frame_active,
            "Object": object_frame_active,
            "Optical": flow_frame_active
        }
        return self.call(f"{agent_id}/frame", Method.GET, data=params)
