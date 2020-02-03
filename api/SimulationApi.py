from api.Api import Api, Method


class SimulationApi(Api):
    @property
    def controller_prefix(self) -> str:
        return "simulation"

    def resume(self):
        return self.call("resume", Method.GET)

    def pause(self):
        return self.call("pause", Method.GET)