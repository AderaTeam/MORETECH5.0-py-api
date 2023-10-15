import numpy as np
from requests import get, Response, Request
from bs4 import BeautifulSoup


def get_rating_by_id(id) -> float:
    url = f'https://yandex.ru/sprav/widget/rating-badge/{id}'
    res = get(url).content.decode('utf-8')
    rate = BeautifulSoup(res).body.find('a', {'class': 'RatingBadgeWidget'}).attrs['style'].split('/')[-2].split('_')[-1]
    return float(rate)


def get_place(
    text: str,
    api_key: str
) -> Response:
    return get(
        f'https://search-maps.yandex.ru/v1/?text={text}&type=biz&lang=ru_RU&apikey={api_key}'
    )


def get_ambit_of_place(
    text: str,
    place_x: str,
    place_y: str,
    ambit_x: str,
    ambit_y: str,
    api_key: str
) -> Response:
    return get(
        f'https://search-maps.yandex.ru/v1/?text={text}&type=biz&lang=ru_RU&apikey={api_key}&ll={place_x},{place_y}&spn={ambit_x},{ambit_y}'
    )

def get_objects_from_ambit_response(r: Response) -> dict:
    return r.json()['features']