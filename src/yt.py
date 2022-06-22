from pytube import YouTube

url = 'https://www.youtube.com/watch?v=sambhEwywVA&list=PL_kCYGU153-0Dz22CDpHpWYD5M3JhKTUW&index=27'
folder_path = 'C:/Users/ngameiro/Videos'

YouTube(url).streams.get_highest_resolution().download(folder_path)