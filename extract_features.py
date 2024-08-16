# Annotation schema for UD entities: https://universaldependencies.org/misc.html#Entity

import pandas as pd
import conllu
import glob
import ast
from doc_splits import gum_test, gum_dev, gum_train, gentle, wsjarrau_test, wsjarrau_dev, wsjarrau_train
# list_obj = ast.literal_eval(list_str)

def parse_entity_instance(token_list, index, ent_id, sent_id, send_id_base_tok_pos, data='gum'):
    # get entity annotations
    parsing_entity = ""
    entities = token_list[0]["misc"]["Entity"].split("(")
    for entity in entities:
        if len(entity) != 0:
            if entity.split("-")[0] == ent_id:
                parsing_entity = entity
    entity_attributes = parsing_entity.split("-")[:6]
    if data == "arrau":
        entity_id, entity_type, coref_type, infostat = int(entity_attributes[0]), entity_attributes[1], \
                                                       entity_attributes[2], entity_attributes[3]
    else:
        entity_id, entity_type, infostat, coref_type = int(entity_attributes[0]), entity_attributes[1], entity_attributes[2], \
                                                   entity_attributes[5].split(")")[0]
    if coref_type == "sgl":
        coref = 0
    else:
        coref = 1

    bridge = 0
    bridge_from_ent = None
    bridge_key = None
    for key in token_list[0]["misc"]:
        if "Bridge" in key:
            bridge_key = key
    if bridge_key:
        bridging_annotations = token_list[0]["misc"][bridge_key].split(",")
        for bridge_anno in bridging_annotations:
            bridge_ent = int(bridge_anno.split("<")[1])
        if bridge_ent == entity_id:
            #print("bridge")
            bridge = 1
            bridge_from_ent = int(bridge_anno.split("<")[0])

    # get entity text
    entity_text = ""
    span = []
    for token in token_list:
        entity_text += token["form"] + " "
        span.append(str(token["sent_id"]) + "-" + str(token["id"]))
    entity_text = entity_text[:-1]

    # get head token
    #multi_inst_count = 0
    head_idx = 0
    token_indexes = []
    token_heads = []
    for token in token_list:
        token_indexes.append(token["id"])
        token_heads.append(token["head"])
    for i in range(len(token_list)):
        if token_heads[i] not in token_indexes and token_heads[i] is not None:
            #if head_idx != 0:
            #    print("Multiple heads")
            #    multi_inst_count += 1
            #    break
            head_idx = i
            break

    head_form, head_lemma, head_deprel, head_xpos, head_id = token_list[head_idx]["form"], token_list[head_idx]["lemma"],\
        token_list[head_idx]["deprel"], token_list[head_idx]["xpos"], token_list[head_idx]["id"]

    head_number = None
    if token_list[head_idx]["feats"] and "Number" in token_list[head_idx]["feats"]:
        head_number = token_list[head_idx]["feats"]["Number"]

    entity_info = {"doc_index": index, "entity_text": entity_text, "entity_id": entity_id, "span": span,
                   "entity_type": entity_type, "infostat": infostat, "coref_type": coref_type, "coref": coref,
                   "bridge": bridge, "bridge_from_ent": bridge_from_ent, "head_form": head_form, "head_lemma": head_lemma,
                   "head_deprel": head_deprel, "head_xpos": head_xpos, "head_number": head_number,
                   "sent_id": sent_id, "head_sent_id": head_id, "head_doc_position": send_id_base_tok_pos + head_id}

    return entity_info #, multi_inst_count


def extract_entity_instances(conllu_sentences, data='gum'):
    instances = []
    open_entities = {}
    #count_tot = 0
    sent_id = 1
    send_id_base_tok_pos = 0
    for sentence in conllu_sentences:
        for token in sentence:
            if "-" in str(token["id"]) or "." in str(token["id"]):
                continue
            token["sent_id"] = sent_id
            if token["misc"] and "Entity" in token["misc"]:
                #print(token["misc"]["Entity"])
                entities = token["misc"]["Entity"].split("(")
                for entity in entities:
                    if len(entity) != 0:
                        if ")" in entity:
                            # closing entity
                            closing_entities = entity.split(")")
                            for closing_ent in closing_entities:
                                if len(closing_ent) != 0:
                                    if "-" in closing_ent:
                                        # single token entity
                                        ent_id = closing_ent.split("-")[0] # entity
                                        entity_info = parse_entity_instance([token], len(instances), ent_id,
                                                                            sent_id, send_id_base_tok_pos, data=data) #, count
                                        #count_tot += count
                                    else:
                                        # multi token entity
                                        ent_id = closing_ent
                                        # parse most recent open entity for this id
                                        entity_info = parse_entity_instance(open_entities[ent_id][-1] + [token],
                                                        len(instances), ent_id, sent_id, send_id_base_tok_pos, data=data) #, count
                                        #count_tot += count
                                        # remove parsed entity
                                        open_entities[ent_id] = open_entities[ent_id][:-1]
                                        # if there are no more open entities, remove from tracker
                                        if len(open_entities[ent_id]) == 0:
                                            open_entities.pop(ent_id)
                                    #if ent_id in open_entities:
                                    #    entity_info = parse_entity_instance(open_entities[ent_id] + [token], len(instances), ent_id)
                                    #    open_entities.pop(ent_id)
                                    #else:
                                    #    entity_info = parse_entity_instance([token], len(instances), ent_id)
                                    instances.append(entity_info)
                        else:
                            # opening entity
                            ent_id = entity.split("-")[0]
                            if ent_id in open_entities:
                                open_entities[ent_id].append([])
                            else:
                                open_entities[ent_id] = [[]]
            for ent_id in open_entities:
                for ent_toks in open_entities[ent_id]:
                    #open_entities[ent_id][] = ent_toks.append(token)
                    ent_toks.append(token)
                #print(entities)
        sent_id += 1
        send_id_base_tok_pos += len(sentence)

    df = pd.DataFrame(instances)
    df['bridge_from_span'] = None
    df['bridge_type'] = None
    # replace None with NA str
    df = df.fillna('N/A')
    return df #, count_tot


def split_annotation(annotation_string):
    annotation_list = []
    seen_ent_ids = []
    if annotation_string == "_":
        return annotation_list
    annotations = annotation_string.split("|")
    for annotation in annotations:
        label, remainder = annotation.split("[")
        ent_id = remainder.split("]")[0]
        while ent_id in seen_ent_ids:
            ent_id += "*"
        seen_ent_ids.append(ent_id)
        annotation_list.append({"ent_id": ent_id, "label": label})

    return annotation_list


def process_tsv_annotation_line(annotations, entities, data='gum'):
    if data == "arrau":
        token_index, token, entity_types, info_stats, coref_types, coref_links = annotations[1], annotations[2], annotations[3], annotations[4], annotations[8], annotations[9]
    else:
        token_index, token, entity_types, info_stats, coref_types, coref_links = annotations[0], annotations[2], annotations[3], annotations[4], annotations[8], annotations[9]

    ent_types_list = split_annotation(entity_types)
    for entity_type in ent_types_list:
        ent_id = entity_type["ent_id"]
        label = entity_type["label"]
        if ent_id in entities:
            # if the current token is adjacent to the previous token for the entity, add it to the ent_text entry
            # and update the span
            #sent_idx, tok_idx = token_index.split("-")
            #last_sent_idx, last_tok_idx = entities[ent_id][-1]["span"][-1].split("-")
            #if sent_idx == last_sent_idx and (int(tok_idx) - int(last_tok_idx)) == 1:
            entities[ent_id]["ent_text"] += " " + token
            entities[ent_id]["span"].append(token_index)
            # else, make a new list entry for a new entity instance
            #else:
            #    entities[ent_id].append({"id": ent_id, "ent_text": token, "ent_type": label, "span": [token_index]})

            if "ent_type" not in entities[ent_id]:
                entities[ent_id]["ent_type"] = label
        else:
            entities[ent_id] = {"id": ent_id, "ent_text": token, "ent_type": label, "span": [token_index]}

    info_stat_list = split_annotation(info_stats)
    for info_stat in info_stat_list:
        ent_id = info_stat["ent_id"]
        label = info_stat["label"]
        if "info_stat" not in entities[ent_id]:
            entities[ent_id]["info_stat"] = label

    coref_types_list = coref_types.split("|")
    coref_links_list = coref_links.split("|")
    for coref_type, coref_link in zip(coref_types_list, coref_links_list):
        if coref_type != "_":
            ent_id_links = coref_link.split("[")[1].split("]")[0]
            link_ent_id, curr_ent_id = ent_id_links.split("_")
            if "bridge" in coref_type:
                #print("tsv bridge")
                if "forward_bridge_ent_id" in entities[curr_ent_id]:
                    entities[curr_ent_id]["forward_bridge_ent_id"].append(link_ent_id)
                    entities[curr_ent_id]["bridge_type"].append(coref_type)
                else:
                    entities[curr_ent_id]["forward_bridge_ent_id"] = [link_ent_id]
                    entities[curr_ent_id]["bridge_type"] = [coref_type]

    return entities


def extract_entities_from_tsv(file_name, data='gum'):
    entities = {}
    with open(file_name, "r") as f:
        file_lines = f.readlines()
    for line in file_lines:
        annotations = line.split("\t")
        if len(annotations) == 11:
            entities = process_tsv_annotation_line(annotations, entities, data=data)

    df = pd.DataFrame(list(entities.values()))

    if "forward_bridge_ent_id" not in df.columns:
        df["forward_bridge_ent_id"] = None

    return df


def compare_and_combine_extracted_entity_instances(conllu_instances, tsv_instances):
    # validate that the extracted entities are the same
    assert len(conllu_instances) == len(tsv_instances)
    # Extract the columns
    conllu_span_values = conllu_instances['span'].apply(sorted)
    tsv_span_values = tsv_instances['span'].apply(sorted)
    # Convert to lists and sort
    sorted_conllu_span = sorted(conllu_span_values.tolist())
    sorted_tsv_span = sorted(tsv_span_values.tolist())
    # Compare the sorted lists
    for (c, s) in zip(sorted_conllu_span, sorted_tsv_span):
        assert c == s
    assert sorted_conllu_span == sorted_tsv_span

    # Combine bridging information from the two dfs
    bridging_instances = conllu_instances[conllu_instances["bridge"] == 1]
    skipped_bridges = 0
    for _, bridge_row in bridging_instances.iterrows():
        bridge_from_ent = bridge_row["bridge_from_ent"] # check this matches in the end
        orig_span = bridge_row["span"]
        # in tsv df, find corresponding row to orig_span
        tsv_anaphor_row = tsv_instances[tsv_instances["span"].apply(lambda x: x == orig_span)]
        #tsv_anaphor_row = tsv_instances[tsv_instances["span"] == orig_span]
        # get the instance id of that row
        if len(tsv_anaphor_row) == 1:
            anaphor_tsv_id = tsv_anaphor_row["id"].item()
        else:
            anaphor_tsv_id = list(tsv_anaphor_row["id"])[0]
        # find the row where that instance id is in the list of the forward bridge column
        tsv_antecedent_row = tsv_instances[tsv_instances["forward_bridge_ent_id"].apply(lambda x: type(x) == list and anaphor_tsv_id in x)]
        #tsv_antecedent_row = tsv_instances[anaphor_tsv_id in tsv_instances["forward_bridge_ent_id"]]
        if tsv_antecedent_row.empty:
            #print("missing antecedent")
            skipped_bridges += 1
            conllu_instances.loc[conllu_instances["span"].apply(lambda x: x == orig_span), "bridge"] = 0
            continue
        # get the span and the bridge_type from that row
        instance_index = tsv_antecedent_row["forward_bridge_ent_id"].item().index(anaphor_tsv_id)
        antecedent_span = tsv_antecedent_row["span"].item()
        bridge_type = tsv_antecedent_row["bridge_type"].item()[instance_index]
        # in conllu df, look up row by that span - the ent id should match the original bridge_from_ent
        conllu_antecedent_row = conllu_instances[conllu_instances["span"].apply(lambda x: x == antecedent_span)]
        #conllu_antecedent_row = conllu_instances[conllu_instances["span"] == antecedent_span]
        if len(conllu_antecedent_row) == 1:
            assert conllu_antecedent_row["entity_id"].item() == bridge_from_ent
        else:
            assert bridge_from_ent in list(conllu_antecedent_row["entity_id"])
        # add span and the bridge_type to the conllu df row for original span
        #df.loc[df['B'] == 'banana', 'A'] = 99
        conllu_instances.loc[conllu_instances["span"].apply(lambda x: x == orig_span), 'bridge_from_span'] = str(antecedent_span)
        conllu_instances.loc[conllu_instances["span"].apply(lambda x: x == orig_span), "bridge_type"] = bridge_type
        #row_to_update = conllu_instances[conllu_instances["span"].apply(lambda x: x == orig_span)]
        #row_to_update.at[0, "bridge_from_span"] = antecedent_span
        #row_to_update.at[0, "bridge_type"] = bridge_type

    conllu_instances["span"] = conllu_instances["span"].apply(lambda x: str(x))

    #print("Time to combine annotations")

    return conllu_instances, skipped_bridges


def make_instance_files(data="gum"):
    if data == "arrau":
        file_list = glob.glob("arrau_conllu/*.conllu")
        #file_list = file_list[:5]
        #file_list = ["arrau_conllu/wsjarrau_1148.conllu"]
    else:
        file_list = glob.glob("dep/*.conllu")
    inst_df = pd.DataFrame()
    pair_df = pd.DataFrame()
    #mult_tot = 0
    skipped_bridges = 0
    #need_to_see = "wsjarrau_1148"
    #seen = False
    for file in file_list:
        #if need_to_see in file:
        #    seen = True
        #if not seen:
        #    continue
        file_name = file.split("/")[-1].split(".")[0]
        print(file_name)
        if data == "arrau":
            genre = file_name.split("_")[0]
        else:
            genre = file_name.split("_")[1]
        with open(file, "r") as f:
            file_text = f.read()
            sentences = conllu.parse(file_text)
        entity_instances = extract_entity_instances(sentences, data=data) #, mult
        #mult_tot += mult
        # add file_name and genre
        entity_instances["doc_id"] = file_name
        entity_instances["genre"] = genre
    #print(mult_tot)

        if data == "arrau":
            tsv_file = "arrau_tsv/" + file_name + ".tsv"
        else:
            tsv_file = "tsv/" + file_name + ".tsv"
        tsv_entities = extract_entities_from_tsv(tsv_file, data=data)

        validated_entity_instances, skipped_bridges_set = compare_and_combine_extracted_entity_instances(entity_instances, tsv_entities)
        skipped_bridges += skipped_bridges_set

        entity_pairs = make_data_pairs(validated_entity_instances)
        inst_df = pd.concat([inst_df, validated_entity_instances], ignore_index=True)
        pair_df = pd.concat([pair_df, entity_pairs], ignore_index=True)
        #break
    bridges = inst_df[inst_df["bridge"] == 1]
    print("Bridging instances:", len(bridges))
    print("Skipped bridges:", skipped_bridges)
    inst_df.to_csv(data + '_entity_instances.csv', sep='\t', index=False)
    pair_df.to_csv(data + '_entity_pairs.csv', sep='\t', index=False)
    return


def join_rows(first_row, second_row):

    doc_id = first_row["doc_id"]
    genre = first_row["genre"]

    bridge = 0
    bridge_type = "N/A"
    coref = 0
    if second_row["bridge"] == 1 and second_row["bridge_from_span"] == first_row["span"]:
        bridge = 1
        bridge_type = second_row["bridge_type"]

    if first_row["coref"] == 1 and second_row["coref"] == 1 and first_row["entity_id"] == second_row["entity_id"]:
        coref = 1

    pair_info = {"doc_id": doc_id, "genre": genre, "t_span": first_row["span"],
            "t_entity_text": first_row["entity_text"], "t_entity_type": first_row["entity_type"],
            "t_infostat": first_row["infostat"], "t_coref_type": first_row["coref_type"],
            "t_head_form": first_row["head_form"], "t_head_lemma": first_row["head_lemma"],
            "t_head_deprel": first_row["head_deprel"], "t_head_xpos": first_row["head_xpos"],
            "t_head_number": first_row["head_number"], "t_sent_id": first_row["sent_id"],
            "t_head_sent_id": first_row["head_sent_id"], "t_head_doc_position": first_row["head_doc_position"],
            "n_span": second_row["span"],
            "n_entity_text": second_row["entity_text"], "n_entity_type": second_row["entity_type"],
            "n_infostat": second_row["infostat"], "n_coref_type": second_row["coref_type"],
            "n_head_form": second_row["head_form"], "n_head_lemma": second_row["head_lemma"],
            "n_head_deprel": second_row["head_deprel"], "n_head_xpos": second_row["head_xpos"],
            "n_head_number": second_row["head_number"], "n_sent_id": second_row["sent_id"],
            "n_head_sent_id": second_row["head_sent_id"], "n_head_doc_position": second_row["head_doc_position"],
            "t_n_entity_type": first_row["entity_type"] + "_" + second_row["entity_type"],
            "t_n_head_lemma": first_row["head_lemma"] + "_" + second_row["head_lemma"],
            "t_n_head_deprel": first_row["head_deprel"] + "_" + second_row["head_deprel"],
            "t_n_head_xpos": first_row["head_xpos"] + "_" + second_row["head_xpos"],
            "t_n_head_number": first_row["head_number"] + "_" + second_row["head_number"],
            "t_n_dist": second_row["head_doc_position"] - first_row["head_doc_position"],
                 "coref": coref, "bridge": bridge, "bridge_type": bridge_type}

    return pair_info


def make_data_pairs(entity_inst_df):
    seen_pairs = set() # doc indexes of instances already paired up
    pairs = []
    bridging_instances = entity_inst_df[entity_inst_df["bridge"] == 1]
    max_bridge_dist = 0
    for _, bridge_row in bridging_instances.iterrows():
        bridge_from_ent_id = bridge_row["bridge_from_ent"]
        bridge_from_span = bridge_row["bridge_from_span"]
        bridge_index = bridge_row["doc_index"]
        # Get row of instance the current bridge is bridged from
        antecedent_row = entity_inst_df.loc[(entity_inst_df["span"] == bridge_from_span) & (entity_inst_df["entity_id"] == bridge_from_ent_id)].squeeze()
        antecedent_doc_index = antecedent_row["doc_index"].item()

        #print(bridge_from_ent_id, bridge_index)
        #bridge_from_ent_id_instances = entity_inst_df[entity_inst_df["entity_id"] == bridge_from_ent_id]
        # I want the rel_ent closest to the bridge_index but not after it to make a pair
        # all other instances of this ent with the bridge inst should be marked as not eligible (mentions)
        #source_ent_indexes = bridge_from_ent_id_instances["doc_index"]
        #closest = None
        #for num in source_ent_indexes:
        #    if num < bridge_index:
        #        closest = num
        #    else:
        #        break
        curr_bridge_dist = bridge_index - antecedent_doc_index
        if curr_bridge_dist > max_bridge_dist:
            max_bridge_dist = curr_bridge_dist
        #print(closest)
        #for _, ent_inst_row in bridge_from_ent_id_instances.iterrows():
        #    doc_idx = ent_inst_row["doc_index"]
        #    if doc_idx == closest:
        #        pair = join_rows(ent_inst_row, bridge_row, closest=True)
        #        pairs.append(pair)
        #        seen_pairs.add((doc_idx, bridge_index))
        pair = join_rows(antecedent_row, bridge_row)
        pairs.append(pair)
        seen_pairs.add((antecedent_doc_index, bridge_index))
    #if max_bridge_dist == 0:
    #    max_bridge_dist = 500
    # get coref and non-coref pairs
    for _, inst_1 in entity_inst_df.iterrows():
        for _, inst_2 in entity_inst_df.iterrows():
            if inst_1["doc_index"] > inst_2["doc_index"] and \
                    (inst_1["doc_index"] - inst_2["doc_index"]) <= max_bridge_dist:
                # Skip anaphor pronouns
                if len(inst_1["entity_text"].split(" ")) == 1 and inst_1["head_xpos"] == "PRP":
                    continue
                # If the pair is unseen, join rows, append, and mark as seen
                if (inst_2["doc_index"], inst_1["doc_index"]) not in seen_pairs:
                    pair = join_rows(inst_2, inst_1)
                    pairs.append(pair)
                    seen_pairs.add((inst_2["doc_index"], inst_1["doc_index"]))

    df = pd.DataFrame(pairs)
    return df


def collapse_entity_types(df):
    # Collapsed types: person, place, organization, concrete, event, time, substance, animate, abstract
    df.loc[df['t_entity_type'] == "space", 't_entity_type'] = "place"
    df.loc[df['n_entity_type'] == "space", 'n_entity_type'] = "place"
    df.loc[df['t_entity_type'] == "object", 't_entity_type'] = "concrete"
    df.loc[df['n_entity_type'] == "object", 'n_entity_type'] = "concrete"
    df.loc[df['t_entity_type'] == "plan", 't_entity_type'] = "event"
    df.loc[df['n_entity_type'] == "plan", 'n_entity_type'] = "event"
    df.loc[df['t_entity_type'] == "medicine", 't_entity_type'] = "substance"
    df.loc[df['n_entity_type'] == "medicine", 'n_entity_type'] = "substance"
    df.loc[df['t_entity_type'] == "animal", 't_entity_type'] = "animate"
    df.loc[df['n_entity_type'] == "animal", 'n_entity_type'] = "animate"
    df.loc[df['t_entity_type'] == "plant", 't_entity_type'] = "concrete"
    df.loc[df['n_entity_type'] == "plant", 'n_entity_type'] = "concrete"
    df.loc[df['t_entity_type'] == "undersp-onto", 't_entity_type'] = "abstract"
    df.loc[df['n_entity_type'] == "undersp-onto", 'n_entity_type'] = "abstract"
    df.loc[df['t_entity_type'] == "disease", 't_entity_type'] = "abstract"
    df.loc[df['n_entity_type'] == "disease", 'n_entity_type'] = "abstract"
    df.loc[df['t_entity_type'] == "numerical", 't_entity_type'] = "abstract"
    df.loc[df['n_entity_type'] == "numerical", 'n_entity_type'] = "abstract"

    df['t_n_entity_type'] = df['t_entity_type'] + "_" + df['n_entity_type']

    return df


def make_data_partition(select_file_list, outfile, balance_strata=False, skip_split_antecedent=False, data="gum"):
    # read main data file
    if data == "arrau":
        df = pd.read_csv("arrau_entity_pairs_wsj.csv", sep='\t')
    else:
        df = pd.read_csv("gum_entity_pairs.csv", sep='\t')
    if skip_split_antecedent and data == "gum":
        # change all bridge_type bridge:aggr to not be bridging
        df.loc[df['bridge_type'] == "bridge:aggr", 'bridge'] = 0

    # collapse entity types for ARRAU and GUM
    df = collapse_entity_types(df)

    # take all instances of bridging from select gum files
    if data == "arrau":
        select_file_list = [file_name.split(".")[0] for file_name in select_file_list]
    select_df = df[df["doc_id"].isin(select_file_list)]
    bridging_pairs = select_df[select_df["bridge"] == 1]
    class_size = len(bridging_pairs)
    print(class_size)
    # take equal number of other coref and non-coref instances
    coref_pairs = select_df[select_df["coref"] == 1]
    non_coref_pairs = select_df[(select_df["bridge"] == 0) & (select_df["coref"] == 0)]
    sample_coref = coref_pairs.sample(n=class_size, random_state=42)
    print(len(sample_coref))
    sample_non_coref = non_coref_pairs.sample(n=class_size, random_state=42)
    print(len(sample_non_coref))
    # Make dats set with balanced strata or with fill distribution
    if balance_strata:
        data = pd.concat([bridging_pairs, sample_coref, sample_non_coref], ignore_index=True)
    else:
        data = pd.concat([bridging_pairs, coref_pairs, non_coref_pairs], ignore_index=True)
    # shuffle
    shuffled_df = data.sample(frac=1)
    # write to file
    shuffled_df.to_csv(outfile, sep='\t', index=False)
    return


def join_data(joint_file, train_file, dev_file):
    train = pd.read_csv(train_file, sep="\t") # "train_gentle_gum_balanced.csv"
    dev = pd.read_csv(dev_file, sep="\t") # "dev_gentle_gum_balanced.csv"
    if 'preds' in dev.columns:
        dev = dev.drop(columns=['preds']) #, 'preds_dist'])
    print(dev.head())
    df = pd.concat([train, dev], ignore_index=True)
    df.to_csv(joint_file, sep='\t', index=False)
    return


def main():
    """
    # Get entities out of tsv file
    test_file = "tsv/GUM_news_warhol.tsv"
    test_entities = extract_entities_from_tsv(test_file)

    with open("dep/GUM_news_warhol.conllu", "r") as f:
        file_text = f.read()
        sentences = conllu.parse(file_text)
    entity_instances = extract_entity_instances(sentences)

    entity_instances["doc_id"] = "GUM_news_warhol"
    entity_instances["genre"] = "news"

    validated_entity_instances = compare_and_combine_extracted_entity_instances(entity_instances, test_entities)

    entity_pairs = make_data_pairs(validated_entity_instances)

    print("end of script")
    """

    #make_instance_files(data="gum")
    #make_instance_files(data="arrau")

    # With GENTLE, balanced strata
    make_data_partition(gum_train + gum_dev, "train_gentle_gum_balanced.csv", balance_strata=True, skip_split_antecedent=True, data="gum")
    make_data_partition(gum_test + gentle, "dev_gentle_gum_balanced.csv", balance_strata=True, skip_split_antecedent=True, data="gum")
    join_data("train_dev_combined_gentle_gum_balanced.tab", "train_gentle_gum_balanced.csv", "dev_gentle_gum_balanced.csv")
    print("Completed: With GENTLE, balanced strata")

    #entity_instances = pd.read_csv("arrau_entity_pairs.csv", sep='\t')
    #b = entity_instances[entity_instances["bridge"]==1]
    #print("x")
    #entity_pairs = make_data_pairs(entity_instances)
    #entity_pairs.to_csv('arrau_entity_pairs.csv', sep='\t', index=False)

    # ARRAU, balanced strata

    make_data_partition(wsjarrau_train + wsjarrau_dev, "train_wsjarrau_balanced.csv", balance_strata=True,
                        skip_split_antecedent=True, data="arrau")
    make_data_partition(wsjarrau_test, "dev_wsjarrau_balanced.csv", balance_strata=True,
                        skip_split_antecedent=True, data="arrau")
    join_data("train_dev_combined_wsjarrau_balanced.tab", "train_wsjarrau_balanced.csv",
              "dev_wsjarrau_balanced.csv")
    print("Completed: ARRAU, balanced strata")

    # Without GENTLE, balanced strata
    #make_data_partition(gum_train, "train_gum_balanced.csv", balance_strata=True)
    #make_data_partition(gum_dev + gum_test, "dev_gum_balanced.csv", balance_strata=True)
    #join_data("train_dev_combined_gum_balanced.tab")
    #print("Completed: Without GENTLE, balanced strata")

    # With GENTLE, original distribution
    #make_data_partition(gum_train + gentle_train, "train_gentle_gum_unbalanced.csv", balance_strata=False)
    #make_data_partition(gum_dev + gum_test + gentle_dev, "dev_gentle_gum_unbalanced.csv", balance_strata=False)
    #join_data("train_dev_combined_gentle_gum_unbalanced.tab")
    #print("Completed: With GENTLE, original distribution")

    # Without GENTLE, original distribution
    #make_data_partition(gum_train, "train_gum_unbalanced.csv", balance_strata=False)
    #make_data_partition(gum_dev + gum_test, "dev_gum_unbalanced.csv", balance_strata=False)
    #join_data("train_dev_combined_gum_unbalanced.tab")
    #print("Completed: Without GENTLE, original distribution")

    return


def remove_non_wsj():
    entity_pairs = pd.read_csv("arrau_entity_pairs.csv", sep='\t')
    wsj_pairs = entity_pairs[(entity_pairs["genre"] == "vpc") | (entity_pairs["genre"] == "wsjarrau")]
    wsj_pairs.to_csv('arrau_entity_pairs_wsj.csv', sep='\t', index=False)

    entity_instances = pd.read_csv("arrau_entity_instances.csv", sep='\t')
    wsj_instances = entity_instances[(entity_instances["genre"] == "vpc") | (entity_instances["genre"] == "wsjarrau")]
    wsj_instances.to_csv('arrau_entity_instances_wsj.csv', sep='\t', index=False)
    return


if __name__ == "__main__":
    #remove_non_wsj()
    main()
