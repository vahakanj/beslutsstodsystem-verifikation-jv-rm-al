'''
Created on 09.04.2018

@author: Iman
@author: Malte
'''

#we make necessary imports
import pandas as pd
import os
import json
from helper import inout

# define the data folders of the dataset and the csv:s we are about to create
SRC_FOLDER = 'data/original/'
TARGET_FOLDER = 'data/data_formatted/'
# where to save the result file
PLAYLISTS_FILE = TARGET_FOLDER + 'playlists.csv'
TRACKS_FILE = TARGET_FOLDER + 'tracks.csv'
ARTISTS_FILE = TARGET_FOLDER + 'artists.csv'
PLAYLISTS_TRACKS_FILE = TARGET_FOLDER + 'playlists_tracks.csv'

# define dictionaries for the data
ri_track = [0]
ri_track_map = {}
ri_artist = [0]
ri_artist_map = {}
ri_pos = [0]

playlists = {}
playlists['playlist_id'] = []
playlists['name'] = []
playlists['num_tracks'] = []
playlists['num_artists'] = []
playlists['num_albums'] = []
playlists['num_followers'] = []
playlists['num_edits'] = []
playlists['duration_ms'] = []
playlists['modified_at'] = []
playlists['collaborative'] = []
playlists['description'] = []

tracks = {}
tracks['track_id'] = []
tracks['track_uri'] = []
tracks['track_name'] = []
tracks['artist_id'] = []
tracks['album_uri'] = []
tracks['duration_ms'] = []
tracks['album_name'] = []

artists = {}
artists['artist_id'] = []
artists['artist_uri'] = []
artists['artist_name'] = []

playlists_tracks = {}
playlists_tracks['playlist_id'] = []
playlists_tracks['track_id'] = []
playlists_tracks['artist_id'] = []
playlists_tracks['pos'] = []

def getid( org, map, ril ):
    if not org in map:
        ril[0] += 1
        map[org] = ril[0]
    return map[org]

if __name__ == '__main__':
    
    results = pd.DataFrame([])
    
    filenames = os.listdir(SRC_FOLDER)
    
    first_track = True
    for i in range(0, len(filenames)):
#   for filename in filenames:
# start crawling through the json files in the data/original folder
        if filenames[i].startswith("mpd.slice.") and filenames[i].endswith(".json"):
            fp = os.sep.join((SRC_FOLDER, filenames[i]))
            josn_str = json.load(open(fp))
            playlists_str = josn_str['playlists']
            
            # load json file in a dataframe
            df_file_playlists = pd.DataFrame.from_dict(playlists_str)
            
            #iterate over playlists
            for index, row in df_file_playlists.iterrows():
                
                playlist_id = row['pid']
                
                #filling the playlists dictionary with data from the json files
                playlists['playlist_id'].append( playlist_id )
                playlists['name'].append( row['name'] )
                playlists['num_tracks'].append( row['num_tracks'] )
                playlists['num_artists'].append( row['num_artists'] )
                playlists['num_albums'].append(row['num_albums'] )
                playlists['num_followers'].append( row['num_followers'] )
                playlists['num_edits'].append( row['num_edits'] )
                playlists['duration_ms'].append( row['duration_ms'] )
                playlists['modified_at'].append( row['modified_at'] )
                playlists['collaborative'].append( row['collaborative'] )
                playlists['description'].append( row['description'] )
                                
                playlist_tracks = row['tracks']
                
                #going through data in the tracks
                for track in playlist_tracks:
                    
                    artist_uri = track['artist_uri']
                    last = ri_artist[0] 
                    artist_id  = getid( artist_uri, ri_artist_map, ri_artist )
                    
                    if artist_id == last + 1:
                                                
                        artists['artist_id'].append( artist_id )
                        artists['artist_uri'].append( artist_uri )
                        artists['artist_name'].append( track['artist_name'] )
                    
                    track_uri = track['track_uri']
                    last = ri_track[0] 
                    track_id = getid( track_uri, ri_track_map, ri_track )
                    pos = track['pos']
                    
                    if track_id == last + 1:
                                                
                        tracks['track_id'].append( track_id )
                        tracks['track_uri'].append( track_uri)
                        tracks['track_name'].append( track['track_name'] )
                        tracks['artist_id'].append( artist_id )
                        tracks['album_uri'].append( track['album_uri'] )
                        tracks['duration_ms'].append( track['duration_ms'] )
                        tracks['album_name'].append( track['album_name'] )
                
                    playlists_tracks['playlist_id'].append( playlist_id )
                    playlists_tracks['track_id'].append( track_id )
                    playlists_tracks['artist_id'].append( artist_id )
                    playlists_tracks['pos'].append( pos )
                    
        
        if (i%10==0):
            print('done for', i, 'files from 1000')
    #we create individual dataframes of playlists, tracks, artists and playlist tracks
    df_playlists = pd.DataFrame.from_dict(playlists)
    df_tracks = pd.DataFrame.from_dict(tracks)
    df_artists = pd.DataFrame.from_dict(artists)
    df_playlists_tracks = pd.DataFrame.from_dict(playlists_tracks)

    #save them to csv files
    print('saving files...')
    inout.ensure_dir( PLAYLISTS_FILE )
    df_playlists.to_csv(PLAYLISTS_FILE, index=False)
    df_tracks.to_csv(TRACKS_FILE, index=False)
    df_artists.to_csv(ARTISTS_FILE, index=False)
    df_playlists_tracks.to_csv(PLAYLISTS_TRACKS_FILE, index=False)
    
    #convert the csv files to binary Feather format for better performance
    inout.convert_feather(TARGET_FOLDER)
    
    