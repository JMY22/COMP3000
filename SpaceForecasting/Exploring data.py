import json
import requests


# I am using this section to explore the data and what I have
# Function to load and explore data from an online JSON file
def load_and_explore_data(url):
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()

        # Explore the structure of the data
        print("Data Structure:")

        if isinstance(data, list):
            # Iterate over items in the list and print their contents
            for item in data[:5]:  # Adjust the number based on the data
                print(item)
        else:
            # Print keys and values of the first few items in a dictionary
            for key, value in list(data.items())[:5]:  # Adjust the number based on the data
                print(f"{key}: {value}")

        # Additional exploration

    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")


# Example usage
if __name__ == "__main__":
    json_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

    # Call the function to load and explore the data
    load_and_explore_data(json_url)
