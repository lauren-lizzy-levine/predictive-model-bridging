import pandas as pd


def make_counts():
    #df1 = pd.read_csv("gum_entity_pairs.csv", sep="\t")
    #df1.loc[df1['bridge_type'] == "bridge:aggr", 'bridge'] = 0
    #df1_bridge = df1[df1["bridge_type"] == "bridge:aggr"]
    #print(len(df1_bridge))

    df = pd.read_csv("train_dev_combined_gentle_gum_balanced.tab", sep="\t")
    count_a = df['bridge_type'].value_counts()
    print(count_a)
    bridge_df = df[df["bridge"] == 1]
    print("Total bridging instances:", len(bridge_df))

    obj_conj_df = df[df["t_n_head_deprel"] == "obj_conj"]
    total = len(obj_conj_df)
    bridge_count = obj_conj_df["bridge"].sum()
    print("\nobj_conj pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    obj_obj_df = df[df["t_n_head_deprel"] == "obj_obj"]
    total = len(obj_obj_df)
    bridge_count = obj_obj_df["bridge"].sum()
    print("obj_obj pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    obl_conj_df = df[df["t_n_head_deprel"] == "obl_conj"]
    total = len(obl_conj_df)
    bridge_count = obl_conj_df["bridge"].sum()
    print("obl_conj pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    object_object_df = df[df["t_n_entity_type"] == "object_object"]
    total = len(object_object_df)
    bridge_count = object_object_df["bridge"].sum()
    print("\nobject_object pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    abstract_abstract_df = df[df["t_n_entity_type"] == "abstract_abstract"]
    total = len(abstract_abstract_df)
    bridge_count = abstract_abstract_df["bridge"].sum()
    print("abstract_abstract pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    time_time_df = df[df["t_n_entity_type"] == "time_time"]
    total = len(time_time_df)
    bridge_count = time_time_df["bridge"].sum()
    print("time_time pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    NNS_NNP_df = df[df["t_n_head_xpos"] == "NNS_NNP"]
    total = len(NNS_NNP_df)
    bridge_count = NNS_NNP_df["bridge"].sum()
    print("\nNNS_NNP pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    NNS_NN_df = df[df["t_n_head_xpos"] == "NNS_NN"]
    total = len(NNS_NN_df)
    bridge_count = NNS_NN_df["bridge"].sum()
    print("NNS_NN pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    NNS_NNS_df = df[df["t_n_head_xpos"] == "NNS_NNS"]
    total = len(NNS_NNS_df)
    bridge_count = NNS_NNS_df["bridge"].sum()
    print("NNS_NNS pairs:")
    print("Total:", total, "Bridge Count:", bridge_count)

    review_df = df[(df["t_n_head_xpos"] == "NNS_NNS") |
                   (df["t_n_head_xpos"] == "NNS_NN") |
                   (df["t_n_head_xpos"] == "NNS_NNP") |
                   (df["t_n_entity_type"] == "time_time") |
                   (df["t_n_entity_type"] == "abstract_abstract") |
                   (df["t_n_entity_type"] == "object_object") |
                   (df["t_n_head_deprel"] == "obl_conj") |
                   (df["t_n_head_deprel"] == "obj_obj") |
                   (df["t_n_head_deprel"] == "obj_conj")]
    review_bridge_df = review_df[review_df["bridge"] == 1]
    print(len(review_bridge_df))


def make_annotation_csv():
    df = pd.read_csv("train_dev_combined_gentle_gum_balanced.tab", sep="\t")
    deprel_combos = ["obl_conj", "obj_obj", "obj_conj"]
    for combo in deprel_combos:
        review_df = df[(df["t_n_head_deprel"] == combo)]
        review_bridge_df = review_df[review_df["bridge"] == 1]
        print(len(review_bridge_df))
        columns = ["doc_id", "t_span", "t_entity_text", "n_span", "n_entity_text", "t_n_head_deprel", "bridge_type"]
        review_bridge_subset_df = review_bridge_df[columns]
        # save to csv file
        review_bridge_subset_df.to_csv(combo + "_bridge_anno.csv", sep=',', index=False)


def main():
    make_annotation_csv()


if __name__ == "__main__":
    main()
