import pandas as pd
import conllu
import glob
from doc_splits import gum_test, gum_dev, gum_train


def parse_entity_instance(token_list, index, ent_id, sent_id, send_id_base_tok_pos):
    # get entity annotations
    parsing_entity = ""
    entities = token_list[0]["misc"]["Entity"].split("(")
    for entity in entities:
        if len(entity) != 0:
            if entity.split("-")[0] == ent_id:
                parsing_entity = entity
    entity_attributes = parsing_entity.split("-")[:6]
    entity_id, entity_type, infostat, coref_type = entity_attributes[0], entity_attributes[1], entity_attributes[2], \
                                                   entity_attributes[5].split(")")[0]
    if coref_type == "sgl":
        coref = 0
    else:
        coref = 1
    if "Bridge" in token_list[0]["misc"]:
        bridge = 1
        bridge_from = token_list[0]["misc"]["Bridge"].split("<")[0]
    else:
        bridge = 0
        bridge_from = None
    # get entity text
    entity_text = ""
    for token in token_list:
        entity_text += token["form"] + " "
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

    entity_info = {"doc_index": index, "entity_text": entity_text, "entity_id": entity_id, "entity_type": entity_type,
                   "infostat": infostat, "coref_type": coref_type, "coref": coref, "bridge": bridge,
                   "bridge_from": bridge_from, "head_form": head_form, "head_lemma": head_lemma,
                   "head_deprel": head_deprel, "head_xpos": head_xpos, "head_number": head_number,
                   "sent_id": sent_id, "head_sent_id": head_id, "head_doc_position": send_id_base_tok_pos + head_id}

    return entity_info #, multi_inst_count


def extract_entity_instances(conllu_sentences):
    instances = []
    open_entities = {}
    #count_tot = 0
    sent_id = 1
    send_id_base_tok_pos = 0
    for sentence in conllu_sentences:
        for token in sentence:
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
                                    if "-" in entity:
                                        # single token entity
                                        ent_id = entity.split("-")[0]
                                        entity_info = parse_entity_instance([token], len(instances), ent_id,
                                                                            sent_id, send_id_base_tok_pos) #, count
                                        #count_tot += count
                                    else:
                                        # multi token entity
                                        ent_id = closing_ent
                                        # parse most recent open entity for this id
                                        entity_info = parse_entity_instance(open_entities[ent_id][-1] + [token],
                                                        len(instances), ent_id, sent_id, send_id_base_tok_pos) #, count
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
    return df #, count_tot


def make_instance_files():
    gum_file_list = glob.glob("dep/*.conllu")
    inst_df = pd.DataFrame()
    pair_df = pd.DataFrame()
    #mult_tot = 0
    for file in gum_file_list:
        file_name = file.split("/")[-1].split(".")[0]
        print(file_name)
        genre = file_name.split("_")[1]
        with open(file, "r") as f:
            file_text = f.read()
            sentences = conllu.parse(file_text)
        entity_instances = extract_entity_instances(sentences) #, mult
        #mult_tot += mult
        # add file_name and genre
        entity_instances["doc_id"] = file_name
        entity_instances["genre"] = genre
    #print(mult_tot)

        entity_pairs = make_data_pairs(entity_instances)
        inst_df = pd.concat([inst_df, entity_instances], ignore_index=True)
        pair_df = pd.concat([pair_df, entity_pairs], ignore_index=True)
        #break
    inst_df.to_csv('gum_entity_instances.csv', sep='\t', index=False)
    pair_df.to_csv('gum_entity_pairs.csv', sep='\t', index=False)
    return


def join_rows(first_row, second_row, closest=False):

    doc_id = first_row["doc_id"]
    genre = first_row["genre"]

    bridge = 0
    coref = 0
    if second_row["bridge"] and second_row["bridge_from"] == first_row["entity_id"] and closest:
        bridge = 1
    if first_row["coref"] and second_row["coref"] and first_row["entity_id"] == second_row["entity_id"]:
        coref = 1

    pair_info = {"doc_id": doc_id, "genre": genre, "t_entity_text": first_row["entity_text"],
                 "t_entity_type": first_row["entity_type"],
            "t_infostat": first_row["infostat"], "t_coref_type": first_row["coref_type"],
            "t_head_form": first_row["head_form"], "t_head_lemma": first_row["head_lemma"],
            "t_head_deprel": first_row["head_deprel"], "t_head_xpos": first_row["head_xpos"],
            "t_head_number": first_row["head_number"], "t_sent_id": first_row["sent_id"],
            "t_head_sent_id": first_row["head_sent_id"], "t_head_doc_position": first_row["head_doc_position"],
            "n_entity_text": second_row["entity_text"], "n_entity_type": second_row["entity_type"],
            "n_infostat": second_row["infostat"], "n_coref_type": second_row["coref_type"],
            "n_head_form": second_row["head_form"], "n_head_lemma": second_row["head_lemma"],
            "n_head_deprel": second_row["head_deprel"], "n_head_xpos": second_row["head_xpos"],
            "n_head_number": second_row["head_number"], "n_sent_id": second_row["sent_id"],
            "n_head_sent_id": second_row["head_sent_id"], "n_head_doc_position": second_row["head_doc_position"],
            "t_n_dist": second_row["head_doc_position"] - first_row["head_doc_position"], "coref": coref, "bridge": bridge}

    return pair_info


def make_data_pairs(entity_inst_df):
    seen_pairs = set() # doc indexes of instances already paired up
    pairs = []
    bridging_instances = entity_inst_df[entity_inst_df["bridge"] == 1]
    for _, bridge_row in bridging_instances.iterrows():
        bridge_from_ent_id = bridge_row["bridge_from"]
        bridge_index = bridge_row["doc_index"]
        #print(bridge_from_ent_id, bridge_index)
        bridge_from_ent_id_instances = entity_inst_df[entity_inst_df["entity_id"] == bridge_from_ent_id]
        # I want the rel_ent closest to the bridge_index but not after it to make a pair
        # all other intances of this ent with the bridge inst should be marked as not eligble
        source_ent_indexes = bridge_from_ent_id_instances["doc_index"]
        closest = None
        for num in source_ent_indexes:
            if num < bridge_index:
                closest = num
            else:
                break
        #print(closest)
        for _, ent_inst_row in bridge_from_ent_id_instances.iterrows():
            doc_idx = ent_inst_row["doc_index"]
            if doc_idx == closest:
                pair = join_rows(ent_inst_row, bridge_row, closest=True)
                pairs.append(pair)
                seen_pairs.add((doc_idx, bridge_index))
    # get coref and non-coref pairs
    for _, inst_1 in entity_inst_df.iterrows():
        for _, inst_2 in entity_inst_df.iterrows():
            if inst_1["doc_index"] > inst_2["doc_index"]:
                if (inst_2["doc_index"], inst_1["doc_index"]) not in seen_pairs:
                    pair = join_rows(inst_2, inst_1)
                    pairs.append(pair)
                    seen_pairs.add((inst_2["doc_index"], inst_1["doc_index"]))

    df = pd.DataFrame(pairs)
    return df


def make_data_partition(select_file_list, outfile):
    # read main data file
    df = pd.read_csv("gum_entity_pairs.csv", sep='\t')
    # take all instances of bridging from select gum files
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
    data = pd.concat([bridging_pairs, sample_coref, sample_non_coref], ignore_index=True)
    # shuffle
    shuffled_df = data.sample(frac=1)
    # write to file
    shuffled_df.to_csv(outfile, sep='\t', index=False)
    return


def join_data():
    train = pd.read_csv("train.csv", sep="\t")
    dev = pd.read_csv("dev.csv", sep="\t")
    dev = dev.drop(columns=['preds', 'preds_dist'])
    print(dev.head())
    df = pd.concat([train, dev], ignore_index=True)
    df.to_csv("train_dev.tab", sep='\t', index=False)
    return


def main():
    make_instance_files()
    make_data_partition(gum_train, "train.csv")
    make_data_partition(gum_dev+gum_test, "dev.csv")
    join_data()
    return


if __name__ == "__main__":
    main()
