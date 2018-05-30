# midi2numpy

# Copyright 2013-2018 James McDermott <jmmcd@jmmcd.net>

# Hereby licensed for use under GPL v2, v3, or any later version at
# your choice, or under MIT or BSD license, at your choice.

# A set of functions for parsing midi files to a simple numerical
# representation for machine learning purposes, also for printing
# simple descriptors etc.

# Uses python-midi available from
# https://github.com/jmmcd/python-midi/ (my fork for py3k)

import sys
import os
from os.path import join, isdir, isfile
from pprint import pformat, pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

import midi

# drum names
names = ["BD", "SD", "CH", "OH", "RD", "CR", "LT", "MT", "HT"]

# map from indices to drum pitches
dm2 = {
    0: 36, # BD
    1: 38, # SD
    2: 42, # CH
    3: 46, # OH
    4: 51, # RD
    5: 49, # CR
    6: 41, # LT
    7: 45, # MT
    8: 48  # HT
    }

# map from drum pitches to indices in our array of drum names. can add
# more elements to this, but map to an existing drum name index
dm = {
    35: 0, 36: 0,                             # BD
    38: 1, 40: 1,                             # SD
    37: 1,                                    # SD (side-stick)
    39: 1,                                    # SD (roll/flam)
    42: 2, 44: 2,                             # CH
    69: 2, 70: 2,                             # CH (shaker)
    46: 3,                                    # OH

    49: 5, 52: 5, 57: 5,                      # CR
    51: 4, 53: 4, 56: 4, 59: 4, 80: 4,        # RD (56 cowbell, 80 mute triangle)
    41: 6, 43: 6, 45: 7, 47: 7, 48: 8, 50: 8, # toms low to high -- LT, MT, HT
    60: 8, 61: 8, 62: 7, 63: 7, 64: 6,        # bongos high to low (use HT, MT, LT)
    67: 8, 68: 7                              # agogos high to low (use HT, MT)
    }

# approx number of steps we are going to quantize into. this assumes a
# 2/4, 4/4, 8/4 time sig.  It's not difficult to deal with different
# time sigs but this is all we need for now.
nsteps = 64
nsteps_per_beat = 2
max_beats = 32

def convert_time(ticks, maxticks, nsteps_this_loop):
    """Given a time in ticks, and the number of ticks in the whole
    pattern, convert it to an array index."""
    # This is a heuristic. A beat can be slightly early or slightly
    # late, so obviously we round to the nearest allowable beat (we
    # allow nsteps of them). But if a hit is actually just intended as
    # a very short beat before an on-beat, it can be mistakenly
    # quantized to that beat. Nothing we can do to prevent that. But
    # it causes a problem if a hit happens just before the end of the
    # pattern, because it will be quantized to the next on-beat, which
    # doesn't exist. So we have to clamp in that case.
    x = int(round(nsteps_this_loop * float(ticks) / maxticks))
    if x >= nsteps_this_loop:
        x = nsteps_this_loop - 1
    return x

def write_image(a, filename):
    """Save an image of the pattern."""
    print(a.shape)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(a, cmap=cm.gray_r, interpolation="nearest")
    ax.tick_params(length=0)
    xticks = range(0, nsteps, 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(map(str, xticks), fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    fig.savefig(filename + ".pdf")
    plt.close()

def numpy2midi(n):
    """Convert a numpy array n to a midi pattern in python-midi
    format, quantised etc."""
    # print(n)
    # print(n.shape)
    resolution = 4
    p = midi.Pattern(resolution=resolution) #, tick_relative=False)

    track = midi.Track() 
    #track = midi.Track(tick_relative=False)

    p.append(track)

    set_tempo = midi.SetTempoEvent()
    set_tempo.set_bpm(120)
    set_tempo.set_mpqn(120)
    track_name = midi.TrackNameEvent(trackname="Track name")
    time_sig = midi.TimeSignatureEvent()
    time_sig.set_numerator(4)
    time_sig.set_denominator(4)
    time_sig.set_metronome(180)
    # time_sig.set_thirtyseconds(32)

    end = midi.EndOfTrackEvent()
    end.tick = nsteps * resolution

    #track.append(set_tempo)
    #track.append(track_name)
    #track.append(time_sig)

    for step in range(nsteps):
        tick = 0
        for drum in range(len(names)):
            # print(n[drum, step])
            if n[drum, step] > 0:
                on = midi.NoteOnEvent()
                on.channel = 9
                on.pitch = dm2[drum]
                on.velocity = int(n[drum, step])
                on.tick = tick
                track.append(on)
                # print(on)
            tick = 0

        tick = resolution
        for drum in range(len(names)):
            if True:
            #elif step > 0 and n[drum, step-1] > 0:
                off = midi.NoteOffEvent()
                off.channel = 9
                off.pitch = dm2[drum]
                off.velocity = 100
                off.tick = tick
                track.append(off)
                # print(off)
            tick = 0

    track.append(end)

    #p.make_ticks_rel()

    return p

def read_midi(ifname):
    if not ifname.endswith(".mid"):
        raise ValueError
    md = midi.read_midifile(ifname)

    # convert from relative to absolute ticks
    md.make_ticks_abs()

    return md

def get_info(md):
    # assume it is a drum track: all notes on same channel, and only a
    # few pitches involved as specified in names and dm. Assume ticks
    # are absolute, not relative.

    info = {}
    info["resolution"] = md.resolution

    for track in md:
        for event in track:
            # print(event)
            if event.name == "Track Name":
                info["track_name"] = event.text
            if event.name == "Set Tempo":
                info["bpm"] = event.get_bpm()
                info["mpqn"] = event.get_mpqn()
            if event.name == "Time Signature":
                info["time_sig_num"] = event.get_numerator()
                info["time_sig_denom"] = event.get_denominator()
            if event.name == "End of Track":
                if True:
                #if event.tick > 0:
                    # sometimes a track is used only for metadata and
                    # has an end of track event with tick=0. this
                    # track's end time is meaningful, so we use it.
                    maxticks = event.tick
                    info["track_end_in_ticks"] = event.tick
    info["track_length_in_beats"] = int(round(maxticks / float(md.resolution)))
    info["track_length_in_ticks"] =  info["track_length_in_beats"] * md.resolution

    return info

def midi2numpy(md):
    """Convert a midi pattern (in python-midi format) to a numpy
    array."""

    # assume it is a drum track: all notes on same channel, and only a
    # few pitches involved as specified in names and dm

    info = get_info(md)

    alen = nsteps_per_beat * info["track_length_in_beats"]

    a = np.zeros((len(names), alen), dtype='int')


    for track in md:
        for event in track:
            if event.name == "Note On" and event.get_velocity() > 0:
                try:
                    idx = dm[event.get_pitch()]
                except:
                    # if the pitch is not in our drum map, ignore it
                    continue
                a[idx,
                  convert_time(event.tick,
                               info["track_length_in_ticks"],
                               alen)] = (
                    event.get_velocity())
    if info["track_length_in_beats"] == 8:
        a = np.concatenate((a, a, a, a), axis=1)
    elif info["track_length_in_beats"] == 16:
        a = np.concatenate((a, a), axis=1)
    else:
        pass # a is already the right length
    assert a.shape[1] == nsteps
    return a

def describe_lib(dirname):
    """For every midi file in the directory or subdirectories, print a
    brief description."""
    # walk the dir
    i = 0
    for root, dirs, files in os.walk(dirname, followlinks=False):

        for file in files:
            # find midi
            if file.endswith(".mid"):
                i += 1
                filepath = join(root, file)
                describe_file(filepath)
    print("Number of MIDI files:", i)
    
def describe_file(filepath):
    print(filepath)
    md = read_midi(filepath)
    # pprint(md)
    info = get_info(md)
    pprint(info)
    print("")

def convert_all_midi_to_numpy(dirname):
    """For every suitable midi file in directory or subdirectories,
    convert to numpy and make an image as well."""
    # walk the dir
    rejected = accepted = 0
    Xs = []
    for root, dirs, files in os.walk(dirname, followlinks=False):

        # if len(Xs) > 50: break # for testing purposes

        for file in files:
            # find midi
            if file.endswith(".mid"):
                filepath = join(root, file)

                # if isfile(filepath.replace(".mid", ".dat")):
                #     print("Will skip", filepath)
                #     continue

                md = read_midi(filepath)
                info = get_info(md)

                # if it's not 4/4, or it's very short or long, skip
                if (info["time_sig_num"] != 4 or
                    (info["track_length_in_beats"] not in (8, 16, 32))):
                    rejected += 1
                    # print("Will not convert", filepath)
                    continue
                # print("Will convert", filepath)
                accepted += 1
                X = midi2numpy(read_midi(filepath))
                write_image(X, filepath.replace(".mid", ".png"))
                Xs.append(X)
            print("rejected", rejected, "accepted", accepted)
    Xs = np.array(Xs)
    print(Xs.shape)
    np.save(dirname.rstrip("/") + ".npy", Xs)



def run_pca(file):
    """Run a principal component analysis on the drum loops. Then test
    it by passing in some points chosen from transformed space -- it
    will give back new arrays which we can transform to a new loop."""

    X = np.load(file)

    n_components = 10

    # run the pca
    pca = PCA(n_components=n_components)
    pca.fit(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))

    # how good are the components?
    print(pca.explained_variance_ratio_)

    # make 10 arbitrary points in the transformed space and see where
    # they go in the X space
    for i in range(10):
        y = np.random.randn(n_components) * 100.0
        x = pca.inverse_transform(y).reshape((len(names), nsteps))
        write_image(x, "tmp%d" % i)

if __name__ == "__main__":
    cmd = sys.argv[1]
    file = sys.argv[2]

    
    if cmd == "describe_file":
        describe_file(file)
    elif cmd == "describe_lib":
        describe_lib(file)
    elif cmd == "convert_lib":
        convert_all_midi_to_numpy(file)
    elif cmd == "run_pca":
        run_pca(file)
    elif cmd == "write_image":
        assert file.endswith(".npy")
        write_image(1.0-np.load(file)[1][0], file.replace(".npy", ""))
        
        