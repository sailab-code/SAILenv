#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# L2S is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import src

from client import Api, Method


class AgentApi(Api):
    """
    TODO: summary ??? Does it miss all the rotation only calls?
    """

    def set_position(self, agent_id: int,
                     x: float, y: float, z: float):
        """
        Set the transform position of the given agent id to the given vector-3 of coordinates x, y, z.

        :param agent_id: the id of the agent
        :param x: x value of the vector-3
        :param y: y value of the vector-3
        :param z: z value of the vector-3
        :return: a json response to the POST call with the given parameters
        """
        return self.call(f"{agent_id}/position", Method.POST, data={"X": x, "Y": y, "Z": z})

    def get_position(self, agent_id: int):
        """
        Get the transform position (a vector-3 of coordinates x, y, z) of the given agent id.

        :param agent_id: the id of the agent
        :return: a json response with the transform position (a vector-3) obtained with GET call
        """
        return self.call(f"{agent_id}/position", Method.GET)

    def set_transform(self, agent_id: int, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
        """
        Set the transform position/rotation of the given agent id to the given vector-3 of coordinates x, y, z.

        :param agent_id: the id of the agent
        :param pos_x: x value of the position vector-3
        :param pos_y: y value of the position vector-3
        :param pos_z: z value of the position vector-3
        :param rot_x: x value of the rotation vector-3
        :param rot_y: y value of the rotation vector-3
        :param rot_z: z value of the rotation vector-3
        :return: a json response to the POST call with the given parameters
        """
        data: {} = {
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
        """
        Get the transform position/rotation of the given agent id.

        :param agent_id: the id of the agent
        :return: a json response to the GET call
        """
        return self.call(f"{agent_id}/transform", Method.GET)

    def register(self):
        """
        Register the agent on the Unity server and assign an id to it.

        :return: a json response to the POST call
        """
        return self.call("register", Method.POST)

    def delete(self, agent_id: int):
        """
        Delete the given agent id on the Unity server.

        :param agent_id: the id of the agent
        :return: a json response to the POST call
        """
        return self.call(f"{agent_id}/delete", Method.POST)

    def get_frame(self,
                  agent_id: int,
                  main_frame_active: bool = True,
                  object_frame_active: bool = True,
                  category_frame_active: bool = True,
                  flow_frame_active: bool = True):
        """
        Get the frames currently rendered in the Unity server, getting the ones defined by the given flags.

        :param agent_id: the id of the agent
        :param main_frame_active: flag for the scene PBR frame
        :param object_frame_active: flag for the object id (color of instances) frame
        :param category_frame_active: flag for the category id (color of categories) frame
        :param flow_frame_active: flag for the optical flow frame
        :return: a json response to the GET call with content the given frames
        """
        params = {
            "Main": main_frame_active,
            "Category": category_frame_active,
            "Object": object_frame_active,
            "Optical": flow_frame_active
        }
        return self.call(f"{agent_id}/frame", Method.GET, data=params)

    @property
    def controller_prefix(self) -> str:
        return "agent"
