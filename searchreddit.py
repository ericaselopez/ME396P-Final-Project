import praw
from psaw import PushshiftAPI
import datetime as dt
import csv

# You must get your own unique client_id, client_secret, and user_agent by registering an application with Reddit
# See the ReadMe on this project's repository for more information
# Paste the appropriate IDs in the empty quotes
r = praw.Reddit(client_id='', client_secret='', user_agent='')
api = PushshiftAPI(r)

# Replace 'placeholder.csv' with the name of the .csv you would like to write to
with open('april20toapril21_redditdata.csv', mode='w', newline='') as csv_file:

    # Writes column headers to file
    headers = 'Date', 'Keyword Count'
    csvwriter = csv.writer(csv_file)
    csvwriter.writerow(headers)

    def get_posts(current_date, subreddit):

        start_date = int(current_date.timestamp()) # fetches epoch value of start date
        next_day = current_date + dt.timedelta(days=1) # adds one day to datetime object
        end_date= int(next_day.timestamp()) # fetches epoch value of end date

        posts = list(api.search_submissions(after=start_date,
                                        before=end_date,
                                        subreddit=subreddit,
                                        q='Tesla|TSLA'))

        comments = list(api.search_comments(after=start_date,
                                        before=end_date,
                                        subreddit=subreddit,
                                        q='Tesla|TSLA'))

        post_count = len(posts)
        comment_count = len(comments)
        total_keywords = post_count + comment_count

        data_row = [(current_date.date(), total_keywords)] # converts datetime to date
        write_to_csv(data_row)

        return

    def write_to_csv(row): # function writes date stamp and keyword count to .csv file
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(row)

        return

    start_date = dt.datetime(2020,4,1) # set this to your start date
    delta = 0

    while delta < 3: # number of days over which to fetch data
        current_date = start_date + dt.timedelta(days=delta)
        get_posts(current_date, 'wallstreetbets') # fetches the current date's posts from subreddit and writes to .csv
        delta += 1
