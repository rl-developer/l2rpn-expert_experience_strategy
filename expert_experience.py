import numpy as np
from grid2op.Agent import BaseAgent
import pandapower as pp
from copy import deepcopy

class ReconnectAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """

    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.nline = 59
        self.ngen = 22
        self.controllablegen = {0, 2, 3, 4, 10, 13, 16, 19, 20, 21}
        self.redispatchable = np.bool([True, False, True, True, True, False, False, False, False, False, True,
                                       False, False, True, False, False, True, False, False, True, True, True])
        self.timestep = 0
        self.target_dispatch = np.zeros(self.ngen)
        self.base_power = [48, 28.2, 0, 150, 50, 0, 0, 0, 0, 0, 47.6, 0, 0, 70, 0, 0, 98.9, 0, 0, 300, 51, 180]
        self.lines_attacked = [0, 9, 13, 14, 18, 23, 27, 39, 45, 56]
        self.lines_cut = set()
        self.thermal_limits = [60.9, 231.9, 272.6, 212.8, 749.2, 332.4, 348., 414.4, 310.1,
                               371.4, 401.2, 124.3, 298.5, 86.4, 213.9, 160.8, 112.2, 291.4,
                               489., 489., 124.6, 196.7, 191.9, 238.4, 174.2, 105.6, 143.7,
                               293.4, 288.9, 107.7, 415.5, 148.2, 124.2, 154.4, 85.9, 106.5,
                               142., 124., 130.2, 86.2, 278.1, 182., 592.1, 173.1, 249.8,
                               441., 344.2, 722.8, 494.6, 494.6, 196.7, 151.8, 263.4, 364.1, 327.]
        self.operationsequence = np.zeros(self.nline)
        self.operationtimestep = np.zeros(self.nline)
        self.operationoption = np.zeros(self.nline)
        self.operationlineaction = np.ones(self.nline) * (-1)
        self.tooperateline = -1

    def line_search(self, observation, action_space):
        new_line_status_array = np.zeros(observation.rho.shape)
        min_rho = max(observation.rho)
        target_lineidx = -1
        target_action = self.action_space(action_space)
        for lineidx in range(self.nline):
            new_line_status_array[lineidx] = -1
            action_space["set_line_status"] = new_line_status_array
            res = self.action_space(action_space)
            obs_, _, done, _ = observation.simulate(res)
            if not done and max(obs_.rho) < min_rho:
                target_lineidx = lineidx
                min_rho = max(obs_.rho)
                target_action = deepcopy(action_space)
            new_line_status_array[lineidx] = 0
        return target_lineidx, min_rho, target_action

    def act(self, observation, reward, done):
        """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
        action_space = {}
        self.timestep += 1
        self.tooperateline = -1

        if np.any(self.operationtimestep > 0):
            idx = np.argmax(self.operationtimestep)
            action_space["set_bus"] = {}
            action_space["set_bus"]["lines_or_id"] = [(32, 1), (34, 1), (37, 1)]
            tmpaction = self.action_space(action_space)
            obs_, _, done, _ = observation.simulate(tmpaction)
            if idx == 31 and self.operationsequence[idx] == 1 and obs_.rho[31] < 0.8 and not done \
                    and observation.time_before_cooldown_sub[23] == 0:
                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                self.tooperateline = 31
            else:
                action_space = {}

        if max(observation.rho) > 1:
            idx = np.argmax(observation.rho)
            if idx == 31 and self.operationsequence[idx] == 0 and observation.line_status[39] and observation.time_before_cooldown_sub[23] == 0:
                action_space["set_bus"] = {}
                action_space["set_bus"]["lines_or_id"] = [(32, 2), (34, 2), (37, 2)]
                print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                self.tooperateline = 31

        lineidx = -1

        if not observation.line_status.all():
            line_disconnected = np.where(observation.line_status == False)[0]
            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] > 0:
                    if idx == 45 or idx == 56:
                        if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(28, 2)]
                            action_space["set_bus"]["generators_id"] = [(9, 2)]
                            action_space["set_bus"]["loads_id"] = [(22, 2)]
                            action_space["redispatch"] = [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(42, 2), (57, 2)]
                            action_space["set_bus"]["generators_id"] = [(16, 2)]
                            action_space["redispatch"] = [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                    if idx == 23:
                        # if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                        #     action_space["set_bus"] = {}
                        #     action_space["set_bus"]["lines_ex_id"] = [(19, 2), (20, 2), (21, 2)]
                        #     action_space["set_bus"]["lines_or_id"] = [(22, 2), (48, 2), (49, 2)]
                        #     action_space["set_bus"]["loads_id"] = [(17, 2)]
                        #     action_space["set_bus"]["generators_id"] = [(7, 2)]
                        #     self.tooperateline = idx
                        #     print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                        #     self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:

                            action_space_t1 = deepcopy(action_space)
                            target_lineidx, min_rho, target_action = self.line_search(observation, action_space_t1)
                            print("Curret Rho: ", max(observation.rho), "Disconnect Line #", target_lineidx, "for Line #", idx, 'Rho: ', min_rho)
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_or_id"] = [(34, 2), (37, 2)]
                            action_space["set_bus"]["generators_id"] = [(12, 2)]
                            action_space_t2 = deepcopy(action_space)
                            obs_, _, done, _ = observation.simulate(self.action_space(action_space_t2))
                            print("Curret Rho: ", max(observation.rho), "Adjust for Line #", idx, 'Rho: ', max(obs_.rho))
                            if max(obs_.rho) > min_rho:
                                action_space = deepcopy(target_action)
                                print(action_space)
                                self.operationoption[idx] = 1
                                self.operationlineaction[idx] = target_lineidx
                            else:
                                self.operationoption[idx] = 0
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                            self.operationsequence[idx] += 1
                    if idx == 14:
                        # if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                        #     action_space["set_bus"] = {}
                        #     action_space["set_bus"]["lines_ex_id"] = [(2, 2)]
                        #     action_space["set_bus"]["lines_or_id"] = [(5, 2)]
                        #     self.tooperateline = idx
                        #     print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                        #     self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space_t1 = deepcopy(action_space)
                            target_lineidx, min_rho, target_action = self.line_search(observation, action_space_t1)
                            print("Curret Rho: ", max(observation.rho), "Disconnect Line #", target_lineidx, "for Line #", idx, 'Rho: ', min_rho)
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(19, 2), (21, 2)]
                            action_space["set_bus"]["lines_or_id"] = [(23, 2), (27, 2), (28, 2), (48, 2), (49, 2), (54, 2)]
                            action_space["set_bus"]["generators_id"] = [(5, 2), (6, 2), (7, 2), (8, 2)]
                            action_space_t2 = deepcopy(action_space)
                            obs_, _, done, _ = observation.simulate(self.action_space(action_space_t2))
                            print("Curret Rho: ", max(observation.rho), "Adjust for Line #", idx, 'Rho: ', max(obs_.rho))
                            if max(obs_.rho) > min_rho:
                                action_space = deepcopy(target_action)
                                print(action_space)
                                self.operationoption[idx] = 1
                                self.operationlineaction[idx] = target_lineidx
                            else:
                                self.operationoption[idx] = 0
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                            self.operationsequence[idx] += 1
                    if idx == 39:
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(31, 2)]
                            action_space["set_bus"]["lines_or_id"] = [(34, 2), (37, 2), (38, 2)]
                            action_space["set_bus"]["generators_id"] = [(11, 2)]
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                    if idx == 27:
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space_t1 = deepcopy(action_space)
                            target_lineidx, min_rho, target_action = self.line_search(observation, action_space_t1)
                            print("Curret Rho: ", max(observation.rho), "Disconnect Line #", target_lineidx,
                                  "for Line #", idx, 'Rho: ', min_rho)
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_or_id"] = [(22, 2), (23, 2), (48, 2), (49, 2), (54, 2)]
                            action_space["set_bus"]["generators_id"] = [(5, 2), (8, 2)]
                            action_space["set_bus"]["loads_id"] = [(17, 2)]
                            action_space_t2 = deepcopy(action_space)
                            obs_, _, done, _ = observation.simulate(self.action_space(action_space_t2))
                            print("Curret Rho: ", max(observation.rho), "Adjust for Line #", idx, 'Rho: ', max(obs_.rho))
                            if max(obs_.rho) > min_rho:
                                action_space = deepcopy(target_action)
                                print(action_space)
                                self.operationoption[idx] = 1
                                self.operationlineaction[idx] = target_lineidx
                            else:
                                self.operationoption[idx] = 0
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                            self.operationsequence[idx] += 1

            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] == 0:
                    if not observation.line_status[23] and idx == self.operationlineaction[23]:
                        continue
                    if not observation.line_status[14] and idx == self.operationlineaction[14]:
                        continue
                    if not observation.line_status[27] and idx == self.operationlineaction[27]:
                        continue
                    lineidx = idx
                    break

        # Recover
        for idx in range(self.nline):
            if observation.time_before_cooldown_line[idx] == 0 and (observation.line_status[idx] or lineidx == idx):
                if idx == 45 or idx == 56:
                    if self.operationsequence[idx] == 1:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(42, 1), (57, 1)]
                        action_space["set_bus"]["generators_id"] = [(16, 1)]
                        action_space["redispatch"] = [(10, -2.8), (13, -2.8), (19, 2.8), (21, 2.8)]
                        self.tooperateline = idx
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                    if self.operationsequence[idx] == 2:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(28, 1)]
                        action_space["set_bus"]["generators_id"] = [(9, 1)]
                        action_space["set_bus"]["loads_id"] = [(22, 1)]
                        action_space["redispatch"] = [(10, -2.8), (13, -2.8), (19, 2.8), (21, 2.8)]
                        self.tooperateline = idx
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                if idx == 23:
                    if self.operationsequence[idx] == 1:
                        if self.operationoption[idx] == 0:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_or_id"] = [(34, 1), (37, 1)]
                            action_space["set_bus"]["generators_id"] = [(12, 1)]
                            self.tooperateline = idx
                            self.operationsequence[idx] -= 1
                            print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                        else:
                            if observation.line_status[idx]:
                                new_line_status_array = np.zeros(observation.rho.shape)
                                new_line_status_array[int(self.operationlineaction[idx])] = 1
                                action_space["set_line_status"] = new_line_status_array
                                self.tooperateline = idx
                                self.operationsequence[idx] -= 1
                                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])

                    # if self.operationsequence[idx] == 2:
                    #     action_space["set_bus"] = {}
                    #     action_space["set_bus"]["lines_ex_id"] = [(19, 1), (20, 1), (21, 1)]
                    #     action_space["set_bus"]["lines_or_id"] = [(22, 1), (48, 1), (49, 1)]
                    #     action_space["set_bus"]["loads_id"] = [(17, 1)]
                    #     action_space["set_bus"]["generators_id"] = [(7, 1)]
                    #     self.tooperateline = idx
                    #     print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                    #     self.operationsequence[idx] -= 1
                if idx == 14:
                    if self.operationsequence[idx] == 1:
                        if self.operationoption[idx] == 0:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(19, 1), (21, 1)]
                            action_space["set_bus"]["lines_or_id"] = [(23, 1), (27, 1), (28, 1), (48, 1), (49, 1), (54, 1)]
                            action_space["set_bus"]["generators_id"] = [(5, 1), (6, 1), (7, 1), (8, 1)]
                            self.tooperateline = idx
                            self.operationsequence[idx] -= 1
                            print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                        else:
                            if observation.line_status[idx]:
                                new_line_status_array = np.zeros(observation.rho.shape)
                                new_line_status_array[int(self.operationlineaction[idx])] = 1
                                action_space["set_line_status"] = new_line_status_array
                                self.tooperateline = idx
                                self.operationsequence[idx] -= 1
                                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])

                    # if self.operationsequence[idx] == 2:
                    #     action_space["set_bus"] = {}
                    #     action_space["set_bus"]["lines_ex_id"] = [(2, 1)]
                    #     action_space["set_bus"]["lines_or_id"] = [(5, 1)]
                    #     self.tooperateline = idx
                    #     print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                    #     self.operationsequence[idx] -= 1
                if idx == 39:
                    if self.operationsequence[idx] == 1:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(31, 1)]
                        action_space["set_bus"]["lines_or_id"] = [(34, 1), (37, 1), (38, 1)]
                        action_space["set_bus"]["generators_id"] = [(11, 1)]
                        self.tooperateline = idx
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                if idx == 27:
                    if self.operationsequence[idx] == 1:
                        if self.operationoption[idx] == 0:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_or_id"] = [(22, 1), (23, 1), (48, 1), (49, 1), (54, 1)]
                            action_space["set_bus"]["generators_id"] = [(5, 1), (8, 1)]
                            action_space["set_bus"]["loads_id"] = [(17, 1)]
                            self.tooperateline = idx
                            self.operationsequence[idx] -= 1
                            print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                        else:
                            if observation.line_status[idx]:
                                new_line_status_array = np.zeros(observation.rho.shape)
                                new_line_status_array[int(self.operationlineaction[idx])] = 1
                                action_space["set_line_status"] = new_line_status_array
                                self.tooperateline = idx
                                self.operationsequence[idx] -= 1
                                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])

        if lineidx != -1:
            new_line_status_array = np.zeros(observation.rho.shape)
            new_line_status_array[lineidx] = 1
            tmpaction = {}
            tmpaction["set_line_status"] = new_line_status_array
            tmpres = self.action_space(tmpaction)
            obs_, _, done, _ = observation.simulate(tmpres)
            # if done or max(obs_.rho) > 1 > max(observation.rho):
            #     print("Can Not Reconnect #", lineidx, 'Estimate Rho:', max(obs_.rho))
            # else:
            action_space["set_line_status"] = new_line_status_array
            print("Reconnect #", lineidx, 'Estimate Rho:', max(obs_.rho))

        if self.tooperateline == 31:
            if self.operationsequence[31] == 1:
                self.operationsequence[31] -= 1
                self.operationtimestep[31] = 0
            else:
                self.operationsequence[31] += 1
                self.operationtimestep[31] = self.timestep


        res = self.action_space(action_space)
        assert res.is_ambiguous()

        return res

    
def make_agent(env, this_directory_path):
    my_agent = ReconnectAgent(env.action_space)
    return my_agent
