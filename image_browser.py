import requests
import random
import time
from io import BytesIO
from PIL import Image

def retrieve_and_display_static_photo():
    """
    Performs a targeted search on Wikimedia Commons for a specific image, 
    retrieves the direct URL, calculates its screen coverage percentage, 
    and opens the static photo in the system's default image viewer.
    """
    API_URL = "https://commons.wikimedia.org/w/api.php"
    
    # 💡 MONITOR RESOLUTION PLUGGED IN 💡
    # Your monitor is 3440px wide x 1440px high.
    MONITOR_WIDTH_PX = 3440  
    MONITOR_HEIGHT_PX = 1440 
    
    # The image search query remains focused on finding a high-quality human portrait.
    specific_search_term = "pretty smiling woman face portrait high resolution"
    generic_search_term = "smiling woman portrait"

    headers = {
        'User-Agent': 'StaticImageFetcherScript/1.0 (Contact: user@example.com)'
    }
    
    def search_wikimedia(query):
        """Helper function to execute the search and return a list of image titles."""
        params_search = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srnamespace": "6",
            "srlimit": "50",
            "origin": "*"
        }
        
        response = requests.get(API_URL, params=params_search, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        image_list = []
        if 'query' in data and 'search' in data['query']:
            for item in data['query']['search']:
                title = item['title']
                if title.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                    image_list.append(title)
        return image_list

    try:
        # --- STEP 1: Search ---
        print(f"1/3: Attempting search for '{specific_search_term}'...")
        image_list = search_wikimedia(specific_search_term)

        if not image_list:
            print(f"   -> Falling back to '{generic_search_term}'...")
            image_list = search_wikimedia(generic_search_term)

        if not image_list:
            print("No valid image files found.")
            return

        random_image_title = random.choice(image_list)
        print(f"2/3: Found file title: {random_image_title}")

        # --- STEP 2: Get the direct file URL (Original Size) ---
        params_info = {
            "action": "query",
            "format": "json",
            "titles": random_image_title, 
            "prop": "imageinfo", 
            "iiprop": "url", 
            "origin": "*"
        }
        
        response = requests.get(API_URL, params=params_info, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        image_url = None
        if 'query' in data and 'pages' in data['query']:
            page_id = next(iter(data['query']['pages']))
            page_data = data['query']['pages'][page_id]
            
            if 'imageinfo' in page_data and len(page_data['imageinfo']) > 0:
                image_url = page_data['imageinfo'][0].get('url')

        if image_url:
            # --- STEP 3: Download, Analyze, and Display ---
            print(f"3/3: Downloading and analyzing image...")
            
            image_response = requests.get(image_url, headers=headers, stream=True)
            image_response.raise_for_status()
            
            image = Image.open(BytesIO(image_response.content))
            
            # --- CALCULATE AND PRINT DIMENSIONS ---
            img_width, img_height = image.size
            
            # Calculate percentage
            width_percent = (img_width / MONITOR_WIDTH_PX) * 100
            height_percent = (img_height / MONITOR_HEIGHT_PX) * 100
            
            # Check for image size relative to monitor
            size_status = ""
            if img_width > MONITOR_WIDTH_PX or img_height > MONITOR_HEIGHT_PX:
                size_status = " (Image is larger than your monitor!)"

            print("\n--- Image Analysis ---")
            print(f"**Image Dimensions:** {img_width} px wide x {img_height} px high")
            print(f"**Monitor Resolution:** {MONITOR_WIDTH_PX} px x {MONITOR_HEIGHT_PX} px")
            print(f"**Coverage:** {width_percent:.2f}% of monitor width, {height_percent:.2f}% of monitor height{size_status}")
            print("----------------------")


            image.show() 
            
            print("\n✅ Successfully displayed static image.")
        else:
            print("Failed to extract the direct image URL from the API response.")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}. Please check your network connection.")
    except ImportError:
        print("\nERROR: The 'Pillow' library is not installed.")
        print("Please run: **pip install Pillow**")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    retrieve_and_display_static_photo()