#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# L2S is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

from client.api import Api, Method


class SimulationApi(Api):
    """
    API for the simulation on the Unity server. It allows to pause and resume the simulation.
    """

    def resume(self):
        """
        Resume the simulation on the Unity server.

        :return: a json response of the GET call
        """
        return self.call("resume", Method.GET)

    def pause(self):
        """
        Pause the simulation on the Unity server.

        :return: a json response of the GET call
        """
        return self.call("pause", Method.GET)

    @property
    def controller_prefix(self) -> str:
        return "simulation"
