import csv
from datetime import timedelta

import pandas as pd

from src.common.utils import Utils


def csv_writer(path='../data/', outfile='clearedPosts', columns=''):
    target_file = open(path + outfile + '.csv', mode='w', encoding='utf-8', newline='\n')
    writer = csv.writer(target_file, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(columns)

    return writer


def get_processed_posts(posts):
    for i, row in posts.iterrows():
        current_message = row['message']

        if current_message is not None and type(current_message) == str:
            cleared_message = Utils.clean_text(current_message, False)
            row['message'] = cleared_message
        else:
            posts.drop(labels=[i], axis=0)

        print(i)

    return posts


def get_coin_info_for_date(coinInfo, date):
    return coinInfo.loc[coinInfo['Date'] == date]['Close'].values[0]


def append_columns(merged, coinInfo):
    for i, row in merged.iterrows():
        row_date = row['Date']
        previous_day = row_date - timedelta(days=1)
        next_day = row_date + timedelta(days=1)
        after_three_days_date = row_date + timedelta(days=3)
        movement_since_previous_day = get_coin_info_for_date(coinInfo, previous_day) - row['Close']
        movement_the_day_after = row['Close'] - get_coin_info_for_date(coinInfo, next_day)
        movement_three_day_after = row['Close'] - get_coin_info_for_date(coinInfo, after_three_days_date)

        merged.loc[i, ['previous_day_closing_price']] = get_coin_info_for_date(coinInfo, previous_day)
        merged.loc[i, ['next_day_closing_price']] = get_coin_info_for_date(coinInfo, next_day)
        merged.loc[i, ['after_three_days_closing_price']] = get_coin_info_for_date(coinInfo, after_three_days_date)
        merged.loc[i, ['movement_since_previous_day']] = movement_since_previous_day
        merged.loc[i, ['movement_the_day_after']] = movement_the_day_after
        merged.loc[i, ['movement_three_days_after']] = movement_three_day_after
        print("Appending columns: " + str(i))

    return merged


def merge_posts_with_coin_info(coin_info, posts):
    merged = posts.merge(coin_info, how='left', left_on='date', right_on='Date')
    merged = append_columns(merged, coin_info)

    return merged


def get_posts(coin_info, group_message_path, coin_names):
    posts = pd.read_json(group_message_path)

    selected_columns = ['id', 'date', 'views', 'message', 'post_author']

    posts = posts[selected_columns]
    processed_posts = get_processed_posts(posts)

    filtered_posts = processed_posts[processed_posts.message.str.contains('|'.join(coin_names), na=False)]
    filtered_posts["date"] = pd.to_datetime(filtered_posts["date"], utc=True).apply(
        lambda t: t.replace(second=0, minute=0, hour=0))

    coin_info["Date"] = pd.to_datetime(coin_info["Date"], utc=True).apply(
        lambda t: t.replace(second=0, minute=0, hour=0))
    print(coin_info["Date"].head())
    print(filtered_posts["date"].head())

    return merge_posts_with_coin_info(coin_info, filtered_posts)


def generate_crypto_coin_model(coin_info_path, result_path, coin_names):
    coin_info = pd.read_csv(coin_info_path)
    binance_data = get_posts(coin_info, '../data/groupMessages/group_messages_binance.json', coin_names)
    bittrex_data = get_posts(coin_info, '../data/groupMessages/group_messages_bittrex.json', coin_names)
    huobi_data = get_posts(coin_info, '../data/groupMessages/group_messages_huobi.json', coin_names)
    kucoin_data = get_posts(coin_info, '../data/groupMessages/group_messages_kucoin.json', coin_names)
    okex_data = get_posts(coin_info, '../data/groupMessages/group_messages_okex.json', coin_names)

    data = pd.concat([binance_data, bittrex_data, huobi_data, kucoin_data, okex_data])
    data.to_csv(result_path)
