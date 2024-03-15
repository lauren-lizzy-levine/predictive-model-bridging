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
    head_token_index = -1
    token_indexes = []
    token_heads = []
    for token in token_list:
        token_indexes.append(token["id"])
        token_heads.append(token["head"])
    for i in range(len(token_list)):
        if token_heads[i] not in token_indexes and token_heads[i] is not None:
            head_token_index = i
            break

    head_form, head_lemma, head_deprel, head_xpos = token_list[i]["form"], token_list[i]["lemma"],\
        token_list[i]["deprel"], token_list[i]["xpos"]

    head_number = None
    if token_list[i]["feats"] and "Number" in token_list[i]["feats"]:
        head_number = token_list[i]["feats"]["Number"]

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


def make_instance_file():
    gum_file_list = glob.glob("dep/*.conllu")
    df = pd.DataFrame()
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
        # print(entity_instances.head()
        # break
        df = pd.concat([df, entity_instances], ignore_index=True)
    df.to_csv('gum_entity_instances.csv', sep='\t')
    return


def make_data_pairs(entity_inst_df):
    return


def main():
    make_instance_file()
    return


if __name__ == "__main__":
    main()
