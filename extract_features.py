import pandas as pd
import conllu
import glob


def parse_entity_instance(token_list, index, ent_id):
    #print(ent_id)
    #print(token_list[0]["misc"]["Entity"])
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
    head_idx = 0
    token_indexes = []
    token_heads = []
    for token in token_list:
        token_indexes.append(token["id"])
        token_heads.append(token["head"])
    for i in range(len(token_list)):
        if token_heads[i] not in token_indexes and token_heads[i] is not None:
            head_idx = i
            break

    head_form, head_lemma, head_deprel, head_xpos = token_list[head_idx]["form"], token_list[head_idx]["lemma"],\
        token_list[head_idx]["deprel"], token_list[head_idx]["xpos"]

    head_number = None
    if token_list[head_idx]["feats"] and "Number" in token_list[head_idx]["feats"]:
        head_number = token_list[head_idx]["feats"]["Number"]

    entity_info = {"doc_index": index, "entity_text": entity_text, "entity_id": entity_id, "entity_type": entity_type,
                   "infostat": infostat, "coref_type": coref_type, "coref": coref, "bridge": bridge,
                   "bridge_from": bridge_from, "head_form": head_form, "head_lemma": head_lemma,
                   "head_deprel": head_deprel, "head_xpos": head_xpos, "head_number": head_number}

    return entity_info


def extract_entity_instances(conllu_sentences):
    instances = []
    open_entities = {}
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
                                        entity_info = parse_entity_instance([token], len(instances), ent_id)
                                    else:
                                        # multi token entity
                                        ent_id = closing_ent
                                        # parse most recent open entity for this id
                                        entity_info = parse_entity_instance(open_entities[ent_id][-1] + [token],
                                                                            len(instances), ent_id)
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

    df = pd.DataFrame(instances)
    return df


def make_instance_files():
    gum_file_list = glob.glob("dep/*.conllu")
    inst_df = pd.DataFrame()
    pair_df = pd.DataFrame()
    for file in gum_file_list:
        file_name = file.split("/")[-1].split(".")[0]
        print(file_name)
        genre = file_name.split("_")[1]
        with open(file, "r") as f:
            file_text = f.read()
            sentences = conllu.parse(file_text)
        entity_instances = extract_entity_instances(sentences)
        # add file_name and genre
        entity_instances["doc_id"] = file_name
        entity_instances["genre"] = genre

        entity_pairs = make_data_pairs(entity_instances)
        # print(entity_instances.head()
        inst_df = pd.concat([inst_df, entity_instances], ignore_index=True)
        pair_df = pd.concat([pair_df, entity_pairs], ignore_index=True)
        #break
    inst_df.to_csv('gum_entity_instances.csv', sep='\t')
    pair_df.to_csv('gum_entity_pairs.csv', sep='\t')
    return


def join_rows(first_row, second_row, closest=False):

    bridge = 0
    coref = 0
    if second_row["bridge"] and second_row["bridge_from"] == first_row["entity_id"] and closest:
        bridge = 1
    if first_row["coref"] and second_row["coref"] and first_row["entity_id"] == second_row["entity_id"]:
        coref = 1

    pair_info = {"fst_entity_text": first_row["entity_text"], "fst_entity_type": first_row["entity_type"],
            "fst_infostat": first_row["infostat"], "fst_coref_type": first_row["coref_type"],
            "fst_head_form": first_row["head_form"], "fst_head_lemma": first_row["head_lemma"],
            "fst_head_deprel": first_row["head_deprel"], "fst_head_xpos": first_row["head_xpos"],
            "fst_head_number": first_row["head_number"],
            "snd_entity_text": second_row["entity_text"], "snd_entity_type": second_row["entity_type"],
            "snd_infostat": second_row["infostat"], "snd_coref_type": second_row["coref_type"],
            "snd_head_form": second_row["head_form"], "snd_head_lemma": second_row["head_lemma"],
            "snd_head_deprel": second_row["head_deprel"], "snd_head_xpos": second_row["head_xpos"],
            "snd_head_number": second_row["head_number"],
            "coref": coref, "bridge": bridge}

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


def main():
    make_instance_files()
    return


if __name__ == "__main__":
    main()
