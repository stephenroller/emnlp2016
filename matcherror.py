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

#------------------------------------------------------------------------------

    #f1_results_csv.append({
    #    'data': DATA_FOLDER_SHORT,
    #    'space': SPACE_FILENAME_SHORT,
    #    'model': model_name,
    #    'features': features,
    #    'dims': space.matrix.shape[1],
    #    'mean': mean,
    #    'std': stderr,
    #    'n_folds': N_FOLDS,
    #    'seed': seed,
    #})

    # make sure we've seen every item

    #print "False Positive Issue:"
    #recalls = defaultdict(list)
    #match_errs = defaultdict(list)
    #for fold in folds:
    #    fake_data = generate_pseudo_data(data, fold[1], 0.5)
    #    if len(fake_data) == 0:
    #        continue
    #    for name, model_name, features in setups:
    #        Xtr, ytr = generate_feature_matrix(data.ix[fold[0]], space, features, global_vocab)
    #        Xte, yte = generate_feature_matrix(fake_data, space, features, global_vocab)

    #        model = model_factory(model_name)
    #        model.fit(Xtr, ytr)

    #        preds = model.predict(Xte)
    #        recalls[(name, model_name, features)].append(metrics.recall_score(yte, preds))
    #        #match_errs[name].append(metrics.recall_score(~yte, preds))
    #        match_errs[(name, model_name, features)].append(float(np.sum(~yte & preds)) / np.sum(~yte))

    #print "%-10s  %-10s   r        m         b      recall   materr" % (" ", " ")
    #for item in recalls.keys():
    #    name, model_name, features = item

    #    R = np.mean(recalls[item])
    #    ME = np.mean(match_errs[item])
    #    #for r, me in zip(recalls[item], match_errs[item], trues[item]):
    #    #    print "%-10s %.3f   %.3f   %.3f" % (item, r, me, t)
    #    m, b, r, p, se = linregress(recalls[item], match_errs[item])
    #    print "%-10s  %-10s  %6.3f   %6.3f    %6.3f  %.3f    %.3f" % (model_name, features, r, m, b, R, ME)

    #    for R, ME in zip(recalls[item], match_errs[item]):
    #        me_results_csv.append({
    #            'name': name,
    #            'model': model_name,
    #            'features': features,
    #            'Features + Data': features + " " + DATA_FOLDER_SHORT,
    #            'recall': R,
    #            'match_error': ME,
    #            'space': SPACE_FILENAME_SHORT,
    #            'data': DATA_FOLDER_SHORT,
    #            'seed': seed,
    #        })




