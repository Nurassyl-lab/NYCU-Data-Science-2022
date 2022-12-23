from utils import *

#before you run this code
#make sure that your sklearn version is 1.2.0

'get dataFrames'
song_names_path = 'link.csv'
val_data_path = 'val_data.csv'

val_data = pd.read_csv(val_data_path)
val_data = val_data.set_index('Index')

song_names = pd.read_csv(song_names_path)
song_names = song_names.set_index('Index')

#This part is unnessary if you have downloaded new csv from my github
#You can delete rows 16~24
#since validation data has only 344 song out of 859, it will be usefull to 
#delete unused song names from {song_names}
rows = np.array(val_data.index)
song_names = song_names.drop(index=song_names.index.difference(rows))

#sort {val_data}
val_data = val_data.sort_values(by = 'Index', ascending = True)


'After dataFrames are ready to use, we will perform INFERENCE'

row, name, scale = get_song(val_data, song_names)

#description of function {get_song}

#when the tetris block starts falling
#you can call this function
#you get row, name, and scale
#using name you can play the song for the player
#using scale you can compare scale that he/she guessed with true scale
#using row you can easily access data in datasets val_data.csv and list.csv
