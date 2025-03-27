import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dqxxbpyba",
    api_key="671636783159459",
    api_secret="bwWzdbvrFrsQu-_Y2zFvYoER0Mc"
)

async def upload_images(files):
    image_urls = []
    for file in files:
        upload_result = cloudinary.uploader.upload(file.file)
        image_urls.append(upload_result["secure_url"])
    return image_urls
