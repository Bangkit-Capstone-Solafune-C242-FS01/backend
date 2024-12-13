import ee
import requests
import uuid

class GEE():
    def __init__(self):
        credentials = ee.ServiceAccountCredentials(None, "gee_credentials.json")
        ee.Initialize(credentials)

    def mask_s2_clouds(self, image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)

    def get_roi_bands(self, long:float, lat:float, radius:int=3000):
        dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterDate('2015-01-01', '2024-12-31')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', .5))
                .map(self.mask_s2_clouds))
        composite = dataset.median()
        bands12 = composite.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"])
        roi = ee.Geometry.Point([long, lat]).buffer(radius).bounds()
        return roi, bands12

    def get_download_url(self, long:float, lat:float, radius:int=3000):
        try:
            roi, bands = self.get_roi_bands(long=long, lat=lat, radius=radius)
            url = bands.getDownloadURL({
                'region': roi.getInfo()['coordinates'],
                'scale': 10,
                'crs': 'EPSG:4326',
                'format': 'GEO_TIFF'
            })
            return url
        except Exception as e:
            return e
    
    def get_tiff_content(self, long:float, lat:float, radius:int=3000, save:bool=True):
        try:
            url = self.get_download_url(long=long, lat=lat, radius=radius)
            res = requests.get(url)
            if save:
                fname = str(uuid.uuid4())
                with open(f"{fname}.tif", "wb") as file:
                    file.write(res.content)
            return res.content
        except Exception as e:
            return e

if __name__ == "__main__":
    gee = GEE()
    print(gee.get_tiff_content(long=139.640264, lat=35.928941, radius=3000)) #japan
