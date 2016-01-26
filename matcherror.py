#def generate_pseudo_data(data, test_fold, percent_fake=0.5):
#    t = data.ix[test_fold]
#    tp = t[t.entails == True]
#    tp1 = list(set(tp.word1))
#    tp2 = list(set(tp.word2))
#    tps = set(zip(tp.word1, tp.word2))
#
#    fps = set()
#    fails = 0
#    while len(fps) < len(tps) / (2 * percent_fake):
#        i = np.random.randint(len(tp1))
#        w1 = tp1[i]
#        j = np.random.randint(len(tp2))
#        w2 = tp2[j]
#
#        if (w1, w2) in tps:
#            fails += 1
#            if fails >= 1000:
#                break
#            continue
#        else:
#            fps.add((w1, w2))
#
#    return pd.DataFrame(
#        [{'word1': w1, 'word2': w2, 'entails': True}
#            for w1, w2 in tps] +
#        [{'word1': w1, 'word2': w2, 'entails': False}
#            for w1, w2 in fps]
#    )


