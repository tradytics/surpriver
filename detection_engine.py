 # Basic libraries
import os
import ta
import sys
import json
import math
import pickle
import random
import requests
import collections
import numpy as np
from os import walk, path
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.stats import linregress
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from data_loader import DataEngine
import warnings

warnings.filterwarnings("ignore")

# Styling for plots
plt.style.use('seaborn-white')
plt.rc('grid', linestyle="dotted", color='#a0a0a0')
plt.rcParams['axes.edgecolor'] = "#04383F"

# Argument parsing
import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument("--top_n", type=int, default = 25, help="How many top predictions do you want to print")
argParser.add_argument("--min_volume", type=int, default = 5000, help="Minimum volume filter. Stocks with average volume of less than this value will be ignored")
argParser.add_argument("--history_to_use", type=int, default = 7, help="How many bars of 1 hour do you want to use for the anomaly detection model.")
argParser.add_argument("--is_load_from_dictionary", type=int, default = 0, help="Whether to load data from dictionary or get it from data source.")
argParser.add_argument("--data_dictionary_path", type=str, default = "dictionaries/data_dictionary.npy", help="Data dictionary path.")
argParser.add_argument("--is_save_dictionary", type=int, default = 1, help="Whether to save data in a dictionary.")
argParser.add_argument("--data_granularity_minutes", type=int, default = 15, help="Minute level data granularity that you want to use. Default is 60 minute bars.")
argParser.add_argument("--is_test", type=int, default = 0, help="Whether to test the tool or just predict for future. When testing, you should set the future_bars to larger than 1.")
argParser.add_argument("--future_bars", type=int, default = 25, help="How many bars to keep for testing purposes.")
argParser.add_argument("--volatility_filter", type=float, default = 0.05, help="Stocks with volatility less than this value will be ignored.")
argParser.add_argument("--output_format", type=str, default = "CLI", help="What format to use for printing/storing results. Can be CLI or JSON.")
argParser.add_argument("--stock_list", type=str, default = "stocks.txt", help="What is the name of the file in the stocks directory which contains the stocks you wish to predict.")
argParser.add_argument("--data_source", type=str, default = "yahoo_finance", help="The name of the data engine to use.")

args = argParser.parse_args()
top_n = args.top_n
min_volume = args.min_volume
history_to_use = args.history_to_use
is_load_from_dictionary = args.is_load_from_dictionary
data_dictionary_path = args.data_dictionary_path
is_save_dictionary = args.is_save_dictionary
data_granularity_minutes = args.data_granularity_minutes
is_test = args.is_test
future_bars = args.future_bars
volatility_filter = args.volatility_filter
output_format = args.output_format.upper()
stock_list = args.stock_list
data_source = args.data_source

"""
Sample run:
python detection_engine.py --is_test 1 --future_bars 25 --top_n 25 --min_volume 5000 --data_granularity_minutes 60 --history_to_use 14 --is_load_from_dictionary 0 --data_dictionary_path 'dictionaries/feature_dict.npy' --is_save_dictionary 1 --output_format 'CLI'
"""

class ArgChecker:
	def __init__(self):
		print("Checking arguments...")
		self.check_arugments()

	def check_arugments(self):
		granularity_constraints_list = [1, 5, 10, 15, 30, 60]
		granularity_constraints_list_string = ''.join(str(value) + "," for value in granularity_constraints_list).strip(",")
		directory_path = str(os.path.dirname(os.path.abspath(__file__)))

		if data_granularity_minutes not in granularity_constraints_list:
			print("You can only choose the following values for 'data_granularity_minutes' argument -> %s\nExiting now..." % granularity_constraints_list_string)
			exit()

		if is_test == 1 and future_bars < 2:
			print("You want to test but the future bars are less than 2. That does not give us enough data to test the model properly. Please use a value larger than 2.\nExiting now...")
			exit()
		
		if output_format not in ["CLI", "JSON"]:
			print("Please choose CLI or JSON for the output format field. Default is CLI.")
			exit()
		if not path.exists(directory_path + f'/stocks/{stock_list}'):
			print("The stocks list file must exist in the stocks directory")
			exit()
		if data_source not in ['binance', 'yahoo_finance']:
			print("Data source must be a valid and supported service.")
			exit()

class Surpriver:
	def __init__(self):
		print("Surpriver has been initialized...")
		self.TOP_PREDICTIONS_TO_PRINT = top_n
		self.HISTORY_TO_USE = history_to_use
		self.MINIMUM_VOLUME = min_volume
		self.IS_LOAD_FROM_DICTIONARY = is_load_from_dictionary
		self.DATA_DICTIONARY_PATH = data_dictionary_path
		self.IS_SAVE_DICTIONARY = is_save_dictionary
		self.DATA_GRANULARITY_MINUTES = data_granularity_minutes
		self.IS_TEST = is_test
		self.FUTURE_BARS_FOR_TESTING = future_bars
		self.VOLATILITY_FILTER = volatility_filter
		self.OUTPUT_FORMAT = output_format
		self.STOCK_LIST = stock_list
		self.DATA_SOURCE = data_source

		# Create data engine
		self.dataEngine = DataEngine(self.HISTORY_TO_USE, self.DATA_GRANULARITY_MINUTES, 
							self.IS_SAVE_DICTIONARY, self.IS_LOAD_FROM_DICTIONARY, self.DATA_DICTIONARY_PATH,
							self.MINIMUM_VOLUME,
							self.IS_TEST, self.FUTURE_BARS_FOR_TESTING,
							self.VOLATILITY_FILTER,
							self.STOCK_LIST,
							self.DATA_SOURCE)
		

	def is_nan(self, object):
		"""
		Checks if a value is null. 
		"""
		return object != object

	def calculate_percentage_change(self, old, new):
		return ((new - old) * 100) / old

	def calculate_return(self, old, new):
		return new / old

	def parse_large_values(self, value):
		if value < 1000:
			value = str(value)
		elif value >= 1000 and value < 1000000:
			value = round(value / 1000, 2)
			value = str(value) + "K"
		else:
			value = round(value / 1000000, 1)
			value = str(value) + "M"

		return value

	def calculate_volume_changes(self, historical_price):
		volume = list(historical_price["Volume"])
		dates = list(historical_price["Datetime"])
		dates = [str(date) for date in dates]

		# Get volume by date
		volume_by_date_dictionary = collections.defaultdict(list)
		for j in range(0, len(volume)):
			date = dates[j].split(" ")[0]
			volume_by_date_dictionary[date].append(volume[j])

		for key in volume_by_date_dictionary:
			volume_by_date_dictionary[key] = np.sum(volume_by_date_dictionary[key]) # taking average as we have multiple bars per day. 

		# Get all dates
		all_dates = list(reversed(sorted(volume_by_date_dictionary.keys())))
		latest_date = all_dates[0]
		latest_data_point =  list(reversed(sorted(dates)))[0]

		# Get volume information
		today_volume = volume_by_date_dictionary[latest_date]
		average_vol_last_five_days = np.mean([volume_by_date_dictionary[date] for date in all_dates[1:6]])
		average_vol_last_twenty_days = np.mean([volume_by_date_dictionary[date] for date in all_dates[1:20]])

		
		return latest_data_point, self.parse_large_values(today_volume), self.parse_large_values(average_vol_last_five_days), self.parse_large_values(average_vol_last_twenty_days)

	def calculate_recent_volatility(self, historical_price):
		close_price = list(historical_price["Close"])
		volatility_five_bars = np.std(close_price[-5:])
		volatility_twenty_bars = np.std(close_price[-20:])
		volatility_all = np.std(close_price)
		return volatility_five_bars, volatility_twenty_bars, volatility_all

	def calculate_future_performance(self, future_data):
		CLOSE_PRICE_INDEX = 4
		price_at_alert = future_data[0][CLOSE_PRICE_INDEX]
		prices_in_future = [item[CLOSE_PRICE_INDEX] for item in future_data[1:]]
		prices_in_future = [item for item in prices_in_future if item != 0]
		total_sum_percentage_change = abs(sum([self.calculate_percentage_change(price_at_alert, next_price) for next_price in prices_in_future]))
		future_volatility = np.std(prices_in_future)
		return total_sum_percentage_change, future_volatility

	def find_anomalies(self):
		"""
		Main function that does everything
		"""

		# Gather data for all stocks
		if self.IS_LOAD_FROM_DICTIONARY == 0:
			features, historical_price_info, future_prices, symbol_names = self.dataEngine.collect_data_for_all_tickers()
		else:
			# Load data from dictionary
			features, historical_price_info, future_prices, symbol_names = self.dataEngine.load_data_from_dictionary()
		
		# Find anomalous stocks using the Isolation Forest model. Read more about the model at -> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
		detector = IsolationForest(n_estimators = 100, random_state = 0)
		detector.fit(features)
		predictions = detector.decision_function(features)
		
		# Print top predictions with some statistics
		predictions_with_output_data = [[predictions[i], symbol_names[i], historical_price_info[i], future_prices[i]] for i in range(0, len(predictions))]
		predictions_with_output_data = list(sorted(predictions_with_output_data))

		#Results object for storing results in JSON format
		results = []

		for item in predictions_with_output_data[:self.TOP_PREDICTIONS_TO_PRINT]:
			# Get some stats to print
			prediction, symbol, historical_price, future_price = item

			# Check if future data is present or not
			if self.IS_TEST == 1 and len(future_price) < 5:
				print("No future data is present. Please make sure that you ran the prior command with is_test enabled or disable that command now. Exiting now...")
				exit()

			latest_date, today_volume, average_vol_last_five_days, average_vol_last_twenty_days = self.calculate_volume_changes(historical_price)
			volatility_vol_last_five_days, volatility_vol_last_twenty_days, _ = self.calculate_recent_volatility(historical_price)
			if average_vol_last_five_days == None or volatility_vol_last_five_days == None:
				continue

			if self.IS_TEST == 0:
				# Not testing so just add/print the predictions
				
				if self.OUTPUT_FORMAT == "CLI":
					print("Last Bar Time: %s\nSymbol: %s\nAnomaly Score: %.3f\nToday Volume: %s\nAverage Volume 5d: %s\nAverage Volume 20d: %s\nVolatility 5bars: %.3f\nVolatility 20bars: %.3f\n----------------------" % 
																	(latest_date, symbol, prediction,
																	today_volume, average_vol_last_five_days, average_vol_last_twenty_days,
																	volatility_vol_last_five_days, volatility_vol_last_twenty_days))
				results.append({
					'latest_date' : latest_date,
					'Symbol' : symbol,
					'Anomaly Score' : prediction,
					'Today Volume' : today_volume,
					'Average Volume 5d' : average_vol_last_five_days,
					'Average Volume 20d' : average_vol_last_twenty_days,
					'Volatility 5bars' : volatility_vol_last_five_days,
					'Volatility 20bars' : volatility_vol_last_twenty_days
				})

			else:
				# Testing so show what happened in the future
				future_abs_sum_percentage_change, _ = self.calculate_future_performance(future_price)

				if self.OUTPUT_FORMAT == "CLI":
					print("Last Bar Time: %s\nSymbol: %s\nAnomaly Score: %.3f\nToday Volume: %s\nAverage Volume 5d: %s\nAverage Volume 20d: %s\nVolatility 5bars: %.3f\nVolatility 20bars: %.3f\nFuture Absolute Sum Price Changes: %.2f\n----------------------" % 
																	(latest_date, symbol, prediction,
																	today_volume, average_vol_last_five_days, average_vol_last_twenty_days,
																	volatility_vol_last_five_days, volatility_vol_last_twenty_days,
																	future_abs_sum_percentage_change))
				results.append({
					'latest_date' : latest_date,
					'Symbol' : symbol,
					'Anomaly Score' : prediction,
					'Today Volume' : today_volume,
					'Average Volume 5d' : average_vol_last_five_days,
					'Average Volume 20d' : average_vol_last_twenty_days,
					'Volatility 5bars' : volatility_vol_last_five_days,
					'Volatility 20bars' : volatility_vol_last_twenty_days,
					'Future Absolute Sum Price Changes' : future_abs_sum_percentage_change
				})

		if self.OUTPUT_FORMAT == "JSON":
			self.store_results(results)

		if self.IS_TEST == 1:
			self.calculate_future_stats(predictions_with_output_data)

	def store_results(self, results):
		"""
		Function for storing results in a file
		"""
		today= dt.datetime.today().strftime('%Y-%m-%d')
		
		prefix = "results"

		if self.IS_TEST != 0:
			prefix = "results_future"

		file_name = '%s_%s.json' % (prefix, str(today))

		#Print results to Result File
		with open(file_name, 'w+') as result_file:
			json.dump(results, result_file)

		print("Results stored successfully in", file_name)

	def calculate_future_stats(self, predictions_with_output_data):
		"""
		Calculate different stats for future data to show whether the anomalous stocks found were actually better than non-anomalous ones
		"""
		future_change = []
		anomalous_score = []
		historical_volatilities = []
		future_volatilities = []

		for item in predictions_with_output_data:
			prediction, symbol, historical_price, future_price = item
			future_sum_percentage_change, future_volatility = self.calculate_future_performance(future_price)
			_, _, historical_volatility = self.calculate_recent_volatility(historical_price)

			# Skip for when there is a reverse split, the yfinance package does not handle that well so percentages get weirdly large
			if abs(future_sum_percentage_change) > 250 or self.is_nan(future_sum_percentage_change) == True or self.is_nan(prediction) == True:
				continue

			future_change.append(future_sum_percentage_change)
			anomalous_score.append(prediction)
			future_volatilities.append(future_volatility)
			historical_volatilities.append(historical_volatility)

		# Calculate correlation and stats
		correlation = np.corrcoef(anomalous_score, future_change)[0, 1]
		anomalous_future_changes = np.mean([future_change[x] for x in range(0, len(future_change)) if anomalous_score[x] < 0]) # Anything less than 0 is considered anomalous
		normal_future_changes = np.mean([future_change[x] for x in range(0, len(future_change)) if anomalous_score[x] >= 0])
		anomalous_future_volatilities = np.mean([future_volatilities[x] for x in range(0, len(future_volatilities)) if anomalous_score[x] < 0]) # Anything less than 0 is considered anomalous
		normal_future_volatilities = np.mean([future_volatilities[x] for x in range(0, len(future_volatilities)) if anomalous_score[x] >= 0])
		anomalous_historical_volatilities = np.mean([historical_volatilities[x] for x in range(0, len(historical_volatilities)) if anomalous_score[x] < 0]) # Anything less than 0 is considered anomalous
		normal_historical_volatilities = np.mean([historical_volatilities[x] for x in range(0, len(historical_volatilities)) if anomalous_score[x] >= 0])
		
		print("\n*************** Future Performance ***************")
		print("Correlation between future absolute change vs anomalous score (lower is better, range = (-1, 1)): **%.2f**\nTotal absolute change in future for Anomalous Stocks: **%.3f**\nTotal absolute change in future for Normal Stocks: **%.3f**\nAverage future volatility of Anomalous Stocks: **%.3f**\nAverage future volatility of Normal Stocks: **%.3f**\nHistorical volatility for Anomalous Stocks: **%.3f**\nHistorical volatility for Normal Stocks: **%.3f**\n" % (
								correlation,
								anomalous_future_changes, normal_future_changes, 
								anomalous_future_volatilities, normal_future_volatilities,
								anomalous_historical_volatilities, normal_historical_volatilities))

		# Plot
		FONT_SIZE = 14
		colors = ['#c91414' if anomalous_score[x] < 0 else '#035AA6' for x in range(0, len(anomalous_score))]
		anomalous_vs_normal = np.array([1 if anomalous_score[x] < 0 else 0 for x in range(0, len(anomalous_score))])
		plt.scatter(np.array(anomalous_score)[anomalous_vs_normal == 1], np.array(future_change)[anomalous_vs_normal == 1], marker='v', color = '#c91414')
		plt.scatter(np.array(anomalous_score)[anomalous_vs_normal == 0], np.array(future_change)[anomalous_vs_normal == 0], marker='P', color = '#035AA6')
		plt.axvline(x = 0, linestyle = '--', color = '#848484')
		plt.xlabel("Anomaly Score", fontsize = FONT_SIZE)
		plt.ylabel("Absolute Future Change", fontsize = FONT_SIZE)
		plt.xticks(fontsize = FONT_SIZE)
		plt.yticks(fontsize = FONT_SIZE)
		plt.legend(["Anomalous", "Normal"], fontsize = FONT_SIZE)
		plt.title("Absolute Future Change", fontsize = FONT_SIZE)
		plt.tight_layout()
		plt.grid()
		plt.show()


# Check arguments
argumentChecker = ArgChecker()

# Create surpriver instance
supriver = Surpriver()

# Generate predictions
supriver.find_anomalies()