#!/usr/bin/env python3

from enum import Enum
import numpy as np
import random
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
from datetime import datetime, timedelta


class Resource:
    def __init__(self, name):
        self.name = name

    def sample_duration(self, activity, simulator):
        raise NotImplementedError
    
    def __str__(self):
        return str(self.name)


class DefaultResource(Resource):
    def __init__(self, name):
        super().__init__(name)
        self.lognormal = lambda m, v : min(np.random.lognormal(np.log(m), v/m),
                                           np.exp(np.log(m) + 3 * v / m)
                                          )

    """
    activated repairs (including the current one)
    """
    def __count_repairs(self, activity):
        previous_activities = activity.instance.activities
        repair_list = list(filter(lambda a : a.type == ActivityTypes.REPAIR,
                             previous_activities))
        return len(repair_list)

    def _sample_repair_duration(self, activity, simulator):
        repairs = self.__count_repairs(activity) # repairs activiated
        match repairs:
            case 1:
                return self.lognormal(3, 1)
            case 2:
                return self.lognormal(8, 0.5)
            case 3:
                return self.lognormal(6, 2)
            case 4 | 5:
                return self.lognormal(9, 1)
            case 6:
                return self.lognormal(7, 1)
        print('obacht')
            

    def _sample_quality_control(self, activity, simulator):
        repairs = self.__count_repairs(activity) # repairs activiated
        match repairs:
            case 1:
                return self.lognormal(2, 0.2)
            case 2:
                return self.lognormal(1, 0.2)
            case _:
                return self.lognormal(0.5, 0.02)


    """
    Simple normal distributed task durations
    """
    def sample_duration(self, activity, simulator):
        match activity.type:
            case ActivityTypes.RECEPTION:
                return self.lognormal(4, 2)
            case ActivityTypes.DISASSEMBLY:
                return self.lognormal(6, 1)
            case ActivityTypes.ACKNOWLEDGEMENT:
                return self.lognormal(5, 3)
            case ActivityTypes.REPAIR:
                return self._sample_repair_duration(activity, simulator)
            case ActivityTypes.QUALITY_CONTROL:
                return self._sample_quality_control(activity, simulator)
            case ActivityTypes.CREATE_INVOICE:
                return self.lognormal(2, 3)
            case ActivityTypes.SHIPPING:
                return self.lognormal(4, 2)


class ActivityTypes(Enum):
    RECEPTION = 0
    ACKNOWLEDGEMENT = 1
    DISASSEMBLY = 2
    REPAIR = 3
    QUALITY_CONTROL = 4
    CREATE_INVOICE = 5
    SHIPPING = 6
    FINISHED = 7


class ActivityState(Enum):
    ACTIVATED = 'SCHEDULE'
    STARTED = 'START'
    COMPLETED = 'COMPLETE'

    def __str__(self):
        return str(self.value)


class Activity:
    def __init__(self, id, type, instance):
        self.id = id
        self.type = type
        self.state = ActivityState.ACTIVATED
        self.instance = instance
        self.resource = None

    def start(self, resource):
        self.state = ActivityState.STARTED
        self.resource = resource

    def complete(self):
        self.state = ActivityState.COMPLETED

    def __str__(self):
        return self.type.name

class ControlFlow:
    def __init__(self, instance):
        self.instance = instance
        iteration_cats = {
            0 : 0.6,
            1 : 0.1,
            2 : 0.2,
            4 : 0.05,
            5 : 0.05
        }
        self.iterations = random.choices(list(iteration_cats.keys()), weights=list(iteration_cats.values()), k=1)[0]
        self.case_conformant = random.choices([True, False], weights=[0.6, 0.4], k=1)[0]
        self.conformance_violation_type = random.choices([0, 1, 2, 3], weights=[0.25, 0.25, 0.25, 0.25], k=1)[0]

    def first_activity_type(self):
        return ActivityTypes.RECEPTION

    def next_activity_types(self, current_activity_type):
        if self.case_conformant:
            match current_activity_type:
                case ActivityTypes.RECEPTION:
                    return [ActivityTypes.DISASSEMBLY, ActivityTypes.ACKNOWLEDGEMENT]
                case ActivityTypes.DISASSEMBLY:
                    acknowledgement = next(filter(lambda a : a.type == ActivityTypes.ACKNOWLEDGEMENT, self.instance.activities))
                    if acknowledgement.state == ActivityState.COMPLETED:
                        return [ActivityTypes.REPAIR]
                    else:
                        return []
                case ActivityTypes.ACKNOWLEDGEMENT:
                    disassembly = next(filter(lambda a : a.type == ActivityTypes.DISASSEMBLY, self.instance.activities))
                    if disassembly.state == ActivityState.COMPLETED:
                        return [ActivityTypes.REPAIR]
                    else:
                        return []
                case ActivityTypes.REPAIR:
                    self.iterations -= 1
                    return [ActivityTypes.QUALITY_CONTROL]
                case ActivityTypes.QUALITY_CONTROL:
                    if self.iterations <= 0:
                        return [ActivityTypes.CREATE_INVOICE]
                    else:
                        return [ActivityTypes.REPAIR]      
                case ActivityTypes.CREATE_INVOICE:
                    return [ActivityTypes.SHIPPING]
                case ActivityTypes.SHIPPING:
                    return [ActivityTypes.FINISHED]
        else:
            # case 1: acknowledge receipt to customer after repair
            if self.conformance_violation_type == 0:
                match current_activity_type:
                    case ActivityTypes.RECEPTION:
                        self.sent_acknowledgement = False
                        return [ActivityTypes.DISASSEMBLY]
                    case ActivityTypes.DISASSEMBLY:
                        return [ActivityTypes.REPAIR]
                    case ActivityTypes.ACKNOWLEDGEMENT:
                        return [ActivityTypes.QUALITY_CONTROL]
                    case ActivityTypes.REPAIR:
                        self.iterations -= 1
                        if not self.sent_acknowledgement:
                            self.sent_acknowledgement = True
                            return [ActivityTypes.ACKNOWLEDGEMENT]
                        else:
                            return [ActivityTypes.QUALITY_CONTROL]
                    case ActivityTypes.QUALITY_CONTROL:
                        if self.iterations <= 0:
                            return [ActivityTypes.CREATE_INVOICE]
                        else:
                            return [ActivityTypes.REPAIR]      
                    case ActivityTypes.CREATE_INVOICE:
                        return [ActivityTypes.SHIPPING]
                    case ActivityTypes.SHIPPING:
                        return [ActivityTypes.FINISHED]
            # case 2: acknowledge receipt to customer after shipping
            elif self.conformance_violation_type == 1:
                match current_activity_type:
                    case ActivityTypes.RECEPTION:
                        return [ActivityTypes.DISASSEMBLY]
                    case ActivityTypes.DISASSEMBLY:
                        return [ActivityTypes.REPAIR]
                    case ActivityTypes.ACKNOWLEDGEMENT:
                        return [ActivityTypes.FINISHED]
                    case ActivityTypes.REPAIR:
                        self.iterations -= 1
                        return [ActivityTypes.QUALITY_CONTROL]
                    case ActivityTypes.QUALITY_CONTROL:
                        if self.iterations <= 0:
                            return [ActivityTypes.CREATE_INVOICE]
                        else:
                            return [ActivityTypes.REPAIR]      
                    case ActivityTypes.CREATE_INVOICE:
                        return [ActivityTypes.SHIPPING]
                    case ActivityTypes.SHIPPING:
                        return [ActivityTypes.ACKNOWLEDGEMENT]
            # case 3: twice repair after each other
            elif self.conformance_violation_type == 2:
                match current_activity_type:
                    case ActivityTypes.RECEPTION:
                        return [ActivityTypes.DISASSEMBLY, ActivityTypes.ACKNOWLEDGEMENT]
                    case ActivityTypes.DISASSEMBLY:
                        acknowledgement = next(filter(lambda a : a.type == ActivityTypes.ACKNOWLEDGEMENT, self.instance.activities))
                        if acknowledgement.state == ActivityState.COMPLETED:
                            return [ActivityTypes.REPAIR]
                        else:
                            return []
                    case ActivityTypes.ACKNOWLEDGEMENT:
                        disassembly = next(filter(lambda a : a.type == ActivityTypes.DISASSEMBLY, self.instance.activities))
                        if disassembly.state == ActivityState.COMPLETED:
                            return [ActivityTypes.REPAIR]
                        else:
                            return []
                    case ActivityTypes.REPAIR:
                        self.iterations -= 1
                        # choose whether another repair or quality control
                        if self.instance.activities[-2].type == ActivityTypes.REPAIR:
                            return [ActivityTypes.QUALITY_CONTROL]
                        else:
                            return [ActivityTypes.REPAIR]
                    case ActivityTypes.QUALITY_CONTROL:
                        if self.iterations <= 0:
                            return [ActivityTypes.CREATE_INVOICE]
                        else:
                            return [ActivityTypes.REPAIR]      
                    case ActivityTypes.CREATE_INVOICE:
                        return [ActivityTypes.SHIPPING]
                    case ActivityTypes.SHIPPING:
                        return [ActivityTypes.FINISHED]
            # case 4:
            elif self.conformance_violation_type == 3:
                match current_activity_type:
                    case ActivityTypes.RECEPTION:
                        self.do_another_repair = True
                        return [ActivityTypes.DISASSEMBLY, ActivityTypes.ACKNOWLEDGEMENT]
                    case ActivityTypes.DISASSEMBLY:
                        acknowledgement = next(filter(lambda a : a.type == ActivityTypes.ACKNOWLEDGEMENT, self.instance.activities))
                        if acknowledgement.state == ActivityState.COMPLETED:
                            return [ActivityTypes.REPAIR]
                        else:
                            return []
                    case ActivityTypes.ACKNOWLEDGEMENT:
                        disassembly = next(filter(lambda a : a.type == ActivityTypes.DISASSEMBLY, self.instance.activities))
                        if disassembly.state == ActivityState.COMPLETED:
                            return [ActivityTypes.REPAIR]
                        else:
                            return []
                    case ActivityTypes.REPAIR:
                        self.iterations -= 1
                        if self.do_another_repair:
                            return [ActivityTypes.QUALITY_CONTROL]
                        else:
                            return [ActivityTypes.CREATE_INVOICE]
                    case ActivityTypes.QUALITY_CONTROL:
                        if self.iterations <= 0:
                            if self.do_another_repair:
                                self.do_another_repair = False
                            return [ActivityTypes.REPAIR]
                        else:
                            return [ActivityTypes.REPAIR]      
                    case ActivityTypes.CREATE_INVOICE:
                        return [ActivityTypes.SHIPPING]
                    case ActivityTypes.SHIPPING:
                        return [ActivityTypes.FINISHED]
        print('obacht')


class ProcessInstance:

    def __init__(self, id, simulator):
        self.id = id
        self.simulator = simulator
        self.control_flow = ControlFlow(self)
        self.current_activity_id = 0
        self.activities = []
        self.resources = []
        self.finished = False

    def start_instance(self):
        first_activity_type = self.control_flow.first_activity_type()
        current_activity = Activity(0, first_activity_type, self)
        self.activities.append(current_activity)
        return [current_activity]

    def get_next_activities(self, completed_activity):
        """
        Gets called every time an activity finishes (EventType.ACTIVITY_COMPLETE)
        """
        if not self.finished:
            next_activity_types = self.control_flow.next_activity_types(completed_activity.type)
            if ActivityTypes.FINISHED in next_activity_types:
                    self.finished = True
            next_activities = []
            for next_activity_type in next_activity_types:
                self.current_activity_id += 1
                next_activity = Activity(self.current_activity_id, next_activity_type, self)
                next_activities.append(next_activity)
                self.activities.append(next_activity)
        return next_activities

    def has_finished(self):
        return self.finished
    
    def __str__(self):
        return str(self.id)


class EventType(Enum):
    INSTANCE_SPAWN = 0
    ACTIVITY_ACTIVATE = 1
    ACTIVITY_START = 2
    ACTIVITY_COMPLETE = 3
    INSTANCE_COMPLETE = 4
    RESOURCE_AVAILABILITY = 5


class Event:
    def __init__(self, event_type, event_time, data):
        self.type = event_type
        self.time = event_time
        self.data = data


class Resources:
    def __init__(self, simulator):
        self.num_total_resources = 50
        self.resources = [
            DefaultResource(i) for i in range(self.num_total_resources)
        ]
        self.idle_resources = self.resources.copy()
        self.working_resources = []
        self.away_resources = []
        self.start_time = datetime(2020, 1, 1)
        self.simulator = simulator
        self.resource_schedule =     {
            0 : [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5,
            1 : [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5,
            2 : [0]*6 + [3, 5, 7, 7, 7, 5, 5, 4, 3, 3, 3, 2, 1] + [0]*5,
            3 : [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5,
            4 : [0]*6 + [4, 5, 7, 7, 7, 7, 7, 2, 3, 2, 0, 0, 0] + [0]*5,
            5 : [0]*24,
            6 : [0]*24
        }
        self.resource_schedule = {k : np.array(v)*5 for k, v in
                                  self.resource_schedule.items()}

    def eligible(self, activity, resource):
        # checks only if resource is not set away
        if resource in self.away_resources:
            return False
        else:
            return True
    
    def change_availability(self, simulator_time):
        weekday = (self.start_time + timedelta(hours=simulator_time)).weekday()
        resource_day_schedule = self.resource_schedule[weekday]
        new_avail_resources = resource_day_schedule[simulator_time % 24]
        last_avail_resources = self.num_total_resources - len(self.away_resources)
        self._change_available_resources(new_avail_resources - last_avail_resources)

    
    def _change_available_resources(self, new_available_resources : int):
        #assert not set(self.away_resources) & set(self.working_resources + self.idle_resources)
        if new_available_resources > 0:
            #make new resources available
            selected = random.sample(self.away_resources, new_available_resources)
            for i in selected:
                self.away_resources.remove(i)
            self.idle_resources.extend(selected)
        elif new_available_resources < 0:
            #remove available resources
            available_resources = list(set(self.resources) - set(self.away_resources))#self.idle_resources + self.working_resources
            selected = random.sample(available_resources, abs(new_available_resources))
            for i in selected:
                if i in self.idle_resources:
                    self.idle_resources.remove(i)
                # removal from working resources happens in free() once current task is completed
            self.away_resources.extend(selected)

    def allocate(self, resource):
        #assert not set(self.away_resources) & set(self.working_resources + self.idle_resources)
        self.working_resources.append(resource)
        self.idle_resources.remove(resource)

    def free(self, resource):
        #assert not set(self.away_resources) & set(self.working_resources + self.idle_resources)
        # resource could have been made unavailable in the meantime
        if resource in self.away_resources:
            self.working_resources.remove(resource)
        else:
            self.idle_resources.append(resource)
            self.working_resources.remove(resource)

    def sample_duration(self, activity, resource):
        return resource.sample_duration(activity, self.simulator)


class ProcessSimulator:
    def __init__(self, start_time = 0, logger = None):
        self.current_time = start_time
        self.max_process_instance_id = -1
        self.event_queue = []
        self.startable_activities = []
        self.logger = logger
        self.resources = Resources(self)
        initial_resource_availability_event = Event(EventType.RESOURCE_AVAILABILITY,
                                                    0, [])
        self.event_queue.append(initial_resource_availability_event)
        self.spawn_instance()
    

    def spawn_instance(self):
        self.max_process_instance_id += 1
        next_instance = ProcessInstance(self.max_process_instance_id, self)
        next_instance_spawn_time = self.current_time + np.random.exponential(scale = 24/5)
        next_instance_spawn_event = Event(EventType.INSTANCE_SPAWN,
                                          next_instance_spawn_time,
                                          {'instance' : next_instance})
        self.event_queue.append(next_instance_spawn_event)

    def activate_activities(self, activities):
        for activity in activities:
            activate_event = Event(EventType.ACTIVITY_ACTIVATE,
                                    self.current_time,
                                    {'activity' : activity})
            self.event_queue.append(activate_event)

    def start_activity(self, activity, resource):
        start_activity = Event(EventType.ACTIVITY_START,
                               self.current_time,
                               {'activity' : activity, 'resource' : resource})
        self.event_queue.append(start_activity)

    def complete_activity(self, activity, resource, activity_completed_time):
        activity_completed = Event(EventType.ACTIVITY_COMPLETE,
                                activity_completed_time,
                                {'activity' : activity, 'resource' : resource})
        assert(activity_completed_time > self.current_time)
        self.event_queue.append(activity_completed)

    def _instance_spawned(self, event):
        # activate first activity
        first_activities = event.data['instance'].start_instance()
        self.activate_activities(first_activities)
        # set next instance spawn
        self.spawn_instance()

    def _activity_activated(self, event):
        self.startable_activities.append(event.data['activity'])

    def _activity_started(self, event):
        activity = event.data['activity']
        resource = event.data['resource']
        # set activity start
        event.data['activity'].start(event.data['resource'])
                    # set activity finish
        duration = self.resources.sample_duration(activity, resource)
        assert(duration > 0)
        self.complete_activity(activity, resource, event.time + duration)

    def _activity_completed(self, event):
        event.data['activity'].complete()
        self.resources.free(event.data['resource'])
        next_activities = event.data['activity'].instance.get_next_activities(event.data['activity'])
        if event.data['activity'].type == ActivityTypes.QUALITY_CONTROL:
            for next_activity in next_activities:
                assert(next_activity.type == ActivityTypes.CREATE_INVOICE or next_activity.type == ActivityTypes.REPAIR )
        if not event.data['activity'].instance.has_finished():
            self.activate_activities(next_activities)

    def _instance_completed(self, event):
        pass

    def _start_activated_activities(self):
        for activity in self.startable_activities[:]:
            random.shuffle(self.resources.idle_resources)
            for resource in self.resources.idle_resources[:]:
                if self.resources.eligible(activity, resource):
                    self.start_activity(activity, resource)
                    self.resources.allocate(resource)
                    self.startable_activities.remove(activity)
                    break

    def _change_resource_availability(self, event : Event):
        self.resources.change_availability(event.time)
        new_resource_availability_event = Event(EventType.RESOURCE_AVAILABILITY,
                                                event.time + 1, [])
        self.event_queue.append(new_resource_availability_event)



    def log_event(self, event):
        if self.logger:
            self.logger.log(event)

    def finish_log(self):
        if self.logger:
            self.logger.finish()

    def simulate(self, max_time = np.inf):
        self.start_time = self.current_time
        while len(self.event_queue):
            if max_time < self.current_time - self.start_time:
                break
            current_event = self.event_queue.pop(0)
            self.current_time = current_event.time

            if current_event.type == EventType.INSTANCE_SPAWN:
                self._instance_spawned(current_event)
            elif current_event.type == EventType.ACTIVITY_ACTIVATE:
                self._activity_activated(current_event)
            elif current_event.type == EventType.ACTIVITY_START:
                self._activity_started(current_event)
            elif current_event.type == EventType.ACTIVITY_COMPLETE:
                self._activity_completed(current_event)
            elif current_event.type == EventType.INSTANCE_COMPLETE:
                self._instance_completed(current_event)
            elif current_event.type == EventType.RESOURCE_AVAILABILITY:
                self._change_resource_availability(current_event)
            
            self._start_activated_activities()
            self.event_queue.sort(key = lambda event : (event.time,
                                                        [EventType.RESOURCE_AVAILABILITY,
                                                         EventType.INSTANCE_COMPLETE,
                                                         EventType.ACTIVITY_COMPLETE,
                                                         EventType.ACTIVITY_START,
                                                         EventType.ACTIVITY_ACTIVATE,
                                                         EventType.INSTANCE_SPAWN].index(event.type)
                                                        ))
            self.log_event(current_event)
        self.finish_log()
        return


class PrintLogger:
    def log(self, event):
        activity = event.data['activity'] if event.type not in [EventType.INSTANCE_COMPLETE, EventType.INSTANCE_SPAWN] else None
        instance = event.data['instance'] if event.type in [EventType.INSTANCE_COMPLETE, EventType.INSTANCE_SPAWN] else event.data['activity'].instance
        resource = event.data['resource'] if event.type in [EventType.ACTIVITY_START] else None
        print(event.time, instance, event.type, activity, resource)

    def finish(self):
        pass

class PandasLogger:
    def __init__(self, out_file):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type == EventType.ACTIVITY_COMPLETE:
            activity = event.data['activity']
            resource = event.data['resource']
            a_id = event.data['activity'].id
            t = event.type
            timestamp = self.start_time + timedelta(hours = event.time)
            self.data.append(['c'+str(activity.instance), str(activity), int(a_id), timestamp, str(resource), str(t)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns = ['case:concept:name', 'concept:name', 'a:id', 'time:timestamp', 'org:resource', 'type'])


class XESLogger(PandasLogger):
    def finish(self):
        super().finish()
        # Convert DataFrame to Event Log
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        # Export to XES
        pm4py.write_xes(log, self.out_file)


class XESLifeCycleLogger:
    def __init__(self, out_file='event_log.xes'):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type in [EventType.ACTIVITY_ACTIVATE, EventType.ACTIVITY_START, EventType.ACTIVITY_COMPLETE]:
            activity = event.data['activity']
            a_id = event.data['activity'].id
            instance = event.data['activity'].instance
            resource = event.data['resource'] if event.type != EventType.ACTIVITY_ACTIVATE else None
            timestamp = self.start_time + timedelta(hours = event.time)
            self.data.append(['c'+str(instance), str(activity), int(a_id), activity.state, timestamp, str(resource)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns = ['case:concept:name', 'concept:name', 'a:id', 'lifecycle:transition', 'time:timestamp', 'org:resource'])
        # Convert DataFrame to Event Log
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        # Export to XES
        pm4py.write_xes(log, self.out_file)

class CSVLogger(PandasLogger):
    def __init__(self, out_file='event_log.csv'):
        super().__init__(out_file)

    def finish(self):
        super().finish()
        self.df.to_csv(self.out_file, index=False)

if __name__ == '__main__':
    '''
    avg cycle time: ~30 hours
    average spawns per day : 5
    average spawns per week : 5*7 = 35
    min working time per week: 30*35 = 1050

    avg working time per working day (mo-fr) : 1050 / 5 = 210

    Base day:
    0-5      6  7  8  9  10 11 12 13 14 15 16 17 18    19-23
    [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5
    = 57 man hours

    Wednesday:
    0-5      6  7  8  9  10 11 12 13 14 15 16 17 18
    [0]*6 + [3, 5, 7, 7, 7, 5, 5, 4, 3, 3, 3, 2, 1] + [0]*5
    = 55 man hours

    Friday:
    0-5      6  7  8  9  10 11 12 13 14 15 16 17 18
    [0]*6 + [4, 5, 7, 7, 7, 7, 7, 2, 3, 2, 0, 0, 0] + [0]*5
    = 51 man hours

    57*3 + 55 + 51 = 277
    personel factor: 277 * 5 = 1385

    base week schedule:
    {
        0 : [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5,
        1 : [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5,
        2 : [0]*6 + [3, 5, 7, 7, 7, 5, 5, 4, 3, 3, 3, 2, 1] + [0]*5,
        3 : [0]*6 + [2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 5, 3, 1] + [0]*5,
        4 : [0]*6 + [4, 5, 7, 7, 7, 7, 7, 2, 3, 2, 0, 0, 0] + [0]*5,
        5 : [0]*24,
        6 : [0]*24
    }
    '''

    #simulator = ProcessSimulator(logger=CSVLogger('test_log.csv'))
    simulator = ProcessSimulator(logger=CSVLogger('repair_shop_event_log_conf_dev.csv'))
    simulator.simulate(2000*24)
