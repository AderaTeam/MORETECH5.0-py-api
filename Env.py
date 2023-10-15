import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch



class Env:
    def __init__(self,
        get_propability_for_checkout_transport: callable,
        change_distance_by_transport: callable,
        get_propability_of_walk_age: callable,
        propability_of_checkout_branch_by_raiting: callable
    ) -> None:
        self.get_propability_for_checkout_transport = get_propability_for_checkout_transport
        self.change_distance_by_transport = change_distance_by_transport
        self.get_propability_of_walk_age = get_propability_of_walk_age
        self.propability_of_checkout_branch_by_raiting = propability_of_checkout_branch_by_raiting

    def step(self, state: dict):
        if state['customer_can_be_processed'] == 0:
            return 0
        auto_check = 0 if state['transport_type'] == 0 else self.get_propability_for_checkout_transport(state['distance'])
        speed = max(25, 55 * state['weather_quality']) if auto_check==1 else 5
        distance = state['distance'] * (1 - self.change_distance_by_transport(state['distance'] / speed, 0.5))
        propability_of_walk_dist_and_age = self.get_propability_of_walk_age(distance=distance, age=state['age'])
        quality_of_district = state['quality_of_district']
        propability_of_check_branch_by_raiting = self.propability_of_checkout_branch_by_raiting(state['rating'])[0]*100
        return propability_of_check_branch_by_raiting * propability_of_walk_dist_and_age


def get_propability_of_walk_age(age: float, distance: float):
    lamb = 1.990e-04
    prop = lamb * np.exp(-lamb * (1*(108-age) + 10*(distance)))
    return prop


def get_propability_for_checkout_transport(distance: float):
    return np.random.choice([0,1], 1, p=[4.7 / distance, 1 - 4.7 / distance])[0]


def change_distance_by_transport(time:float, city_size: float):
    '''
        time [km/s]
        city_size [km]
    '''
    return (-4 * city_size * time * np.log(time) - 6 * city_size + 4 * city_size * time * np.log(3 + 2 * time))/(3 * time)


def from_vector_to_dict(vector: iter):
    return {
        'type_of_customer': vector[0],
        'age': vector[1],
        'transport_type': vector[2],
        'disability': vector[3],
        'rating': vector[4],
        'customer_can_be_processed': vector[5],
        'distance': vector[6],
        'loading_of_branch': vector[7],
        'speed_of_execution': vector[8],
        'quality_of_district': vector[9],
        'weather_quality': vector[10],
    }


def dataset_preprocessor_for_raiting(raiting: list, pf: PolynomialFeatures, stdscaler = StandardScaler):
    d = pd.DataFrame({'Rating': raiting, 'Reviews': np.ones(len(raiting)), 'Installs': np.ones(len(raiting))})
    d = pd.DataFrame({'Rating': stdscaler.transform(d)[:, 0], 'help': np.ones(len(raiting))})
    return pf.transform(d)


def model_pred(model: torch.nn.Module, loss, optimizer, data, from_vector_to_dict: callable, env, epochs=50):
    for _ in range(epochs):
        for x_b in data[torch.randperm(data.shape[0])]:
            outputs = model(x_b)
            s = env.step(from_vector_to_dict(x_b.numpy()))
            loss_value = loss(outputs, 2**s)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
    print( model(data).detach())
    return model(data).detach().numpy()


class FastModel(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(FastModel, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, n_observations*3)
        self.layer2 = torch.nn.Linear(n_observations*3, int(n_observations*1))
        self.layer3 = torch.nn.Linear(int(n_observations*1), n_actions)
        
    def forward(self, x):
        x = torch.nn.functional.tanh(self.layer1(x))
        x = torch.nn.functional.tanh(self.layer2(x))
        return self.layer3(x)
    
