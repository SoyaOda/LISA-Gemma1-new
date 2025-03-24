# vlpartデータセットの修正手順

## 検証結果のまとめ

vlpartデータセットのチェックを行った結果、次の点が判明しました：

1. **PACOデータセット**：利用可能
   - パスの違い：
     - オリジナルの期待パス：`vlpart/paco/annotations/paco_lvis_v1_train.json`
     - 現在の実際のパス：`vlpart/paco/annotations/paco_lvis_v1", "paco_lvis_v1_train.json`

2. **Pascal Partデータセット**：利用可能
   - いくつかのアノテーションに空のセグメンテーションがあるため、マスク生成時の処理が必要

## 修正が必要なファイル

1. `utils/sem_seg_dataset.py`
2. 必要に応じて、他のデータセット処理ファイル

## 修正内容

### 1. `init_paco_lvis` 関数の修正

```python
def init_paco_lvis(base_image_dir):
    # オリジナルのパスをチェック
    original_path = os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json")
    # 現在のパスをチェック
    current_path = os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json")
    
    # 存在するパスを使用
    if os.path.exists(original_path):
        coco_path = original_path
    elif os.path.exists(current_path):
        coco_path = current_path
    else:
        raise FileNotFoundError('paco_lvis_v1_train.jsonが見つかりません')
    
    coco_api_paco_lvis = COCO(coco_path)
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis
```

### 2. マスク生成とアノテーション処理の修正

SemSegDatasetクラスの__getitem__メソッド内でマスク生成を行う部分を修正します：

```python
# ... 既存のコード ...
if ds in ["paco_lvis", "pascal_part"]:
    masks = []
    for ann in sampled_anns:
        try:
            # アノテーションの構造を確認
            if "segmentation" not in ann or not ann["segmentation"]:
                # 空のマスクを作成
                mask_img = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                masks.append(mask_img)
                continue
            
            # マスクを生成
            m = coco_api.annToMask(ann)
            masks.append(m)
        except Exception as e:
            print(f"マスク生成エラー: {e}, アノテーション: {ann}")
            # 空のマスクで代替
            mask_img = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            masks.append(mask_img)
    
    if not masks:  # マスクが生成できなかった場合は別のサンプルを取得
        return self.__getitem__(0)
    
    masks = np.stack(masks, axis=0)
    masks = torch.from_numpy(masks)
    label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
# ... 既存のコード ...
```

## COCO APIのannToMask方法の改善

必要に応じて、utils/helper.pyなどに以下のような独自のマスク生成関数を追加することも検討できます：

```python
def safe_ann_to_mask(ann, height, width):
    """
    アノテーションからマスクを安全に生成するヘルパー関数
    """
    from pycocotools import mask as mask_utils
    
    if "segmentation" not in ann or not ann["segmentation"]:
        return np.zeros((height, width), dtype=np.uint8)
    
    segm = ann["segmentation"]
    
    if isinstance(segm, list):
        if len(segm) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        elif isinstance(segm[0], list):
            # ポリゴン形式（リストのリスト）
            rle = mask_utils.frPyObjects(segm, height, width)
        else:
            print(f"不明なセグメンテーション形式: {segm}")
            if "bbox" in ann:
                # バウンディングボックスからマスクを作成
                bbox = ann["bbox"]
                mask = np.zeros((height, width), dtype=np.uint8)
                x, y, w, h = [int(v) for v in bbox]
                mask[y:y+h, x:x+w] = 1
                return mask
            return np.zeros((height, width), dtype=np.uint8)
    elif isinstance(segm, dict):
        # RLE形式をそのまま使用
        rle = [segm]
    else:
        print(f"不明なセグメンテーション型: {type(segm)}")
        return np.zeros((height, width), dtype=np.uint8)
    
    # RLEをマスクにデコード
    for i in range(len(rle)):
        if "counts" in rle[i] and not isinstance(rle[i]["counts"], bytes):
            rle[i]["counts"] = rle[i]["counts"].encode()
    
    mask = mask_utils.decode(rle)
    if len(mask.shape) > 2:
        mask = np.sum(mask, axis=2)
    mask = mask.astype(np.uint8)
    
    return mask
```

## オリジナルLISAコードとの違いを考慮した学習用データセット修正

1. データパスの違いに対応
2. アノテーション構造の違いに対応
3. マスク生成エラーへの対応

これらの修正を行うことで、現在のvlpartフォルダ構造でもオリジナルのLISAコードと同様に学習が可能になります。 