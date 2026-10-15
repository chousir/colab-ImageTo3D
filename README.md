# colab-ImageTo3D

使用 Google Colab 免費版 T4，把 Drive 裡的一批正面圖自動做成貼好圖的 3D 模型（GLB）：MV-Adapter 補出側面與背面，Hunyuan3D-2mv 建形狀，Hunyuan3D-2 上貼圖。輸出格式符合 [RagnarokChange](https://github.com/chousir/RagnarokChange) 的素材投遞規則。Notebook：[`hunyuan3d2_colab_t4.ipynb`](hunyuan3d2_colab_t4.ipynb)

另有 **Kaggle 2×T4 高精度版** [`hunyuan3d2_kaggle_2xt4.ipynb`](hunyuan3d2_kaggle_2xt4.ipynb)：兩張 GPU 同時做、所有類別都用最高設定、可以在背景跑 12 小時，見[下方說明](#kaggle-版2t4高精度)。

## 固定的 Colab 環境與套件版本

在 Colab 選「執行階段 → 變更執行階段類型」：硬體加速器選 **T4 GPU**，**執行階段版本選 `2026.07`**。Colab 預設的最新版已改用 Python 3.13，部分套件（例如 `xatlas`）沒有 3.13 的預編譯 wheel，所以 Notebook 固定使用 2026.07。區塊 1 會檢查版本，不符時直接停止。

| 項目 | 固定版本 |
|---|---|
| Colab runtime | `2026.07`（Python 3.12.13、PyTorch 2.11.0、NumPy 2.0.2，沿用 Colab 內建，不重裝） |
| GPU | T4 16 GB（免費版），系統 RAM 約 12.7 GB |
| Hunyuan3D-2 原始碼 | commit `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| MV-Adapter 原始碼 | commit `4277e0018232bac82bb2c103caf0893cedb711be` |
| Hugging Face 套件 | `transformers==4.57.3`、`diffusers==0.34.0`、`huggingface_hub==0.36.0`、`accelerate==1.10.1`、`peft==0.17.1` |
| 3D／網格套件 | `pymeshlab==2025.7.post1`、`xatlas==0.0.11`、`trimesh==4.8.3`、`pygltflib==1.16.5`、`nvdiffrast` v0.3.3 |
| 去背 | `rembg==2.0.67`、`pymatting==1.1.14`、`onnxruntime==1.22.1` |
| 其他 | `omegaconf==2.3.0`、`einops==0.8.1`、`pybind11==3.0.1`、`ninja==1.13.0`、`pooch==1.8.2`、`jsonschema==4.25.1`、`jaxtyping==0.3.2`、`typeguard==4.4.4` |

完整清單在區塊 3 的 `PINNED`。安裝時會以 Colab 現有的 torch、numpy、scipy、opencv 等版本作為 constraints，避免把它們升級而弄壞 CUDA 擴充。Colab 只會保留舊 runtime 版本一段時間；若 `2026.07` 從選單消失，需改用新版 runtime，並同步更新區塊 1 的版本檢查與區塊 3 的 `PINNED`。

## 使用方式

1. 在 Colab 開啟 Notebook，「執行階段 → 變更執行階段類型」：T4 GPU，執行階段版本 `2026.07`。
2. 把圖放進 `MyDrive/RagnarokChange/hunyuan3d/in/`。檔名是 `<類別>-<id>-<front|stuff>.png`，id 用英文小寫、數字、底線；**第一個字決定生成方式**：

   | 類別 | 例子 | 生成方式 | 面數上限 |
   |---|---|---|---|
   | `character` | `character-novice_female-front.png` | 正面圖補出左、背、右視角 → Hunyuan3D-2mv；光頭模式；縮放到身高 1.65 m | 60,000 |
   | `monster` | `monster-gelly-front.png` ＋ `monster-gelly-back.png` | **正面、背面兩張都由你提供**（不生成視角）→ Hunyuan3D-2mv；不加光頭提示詞、不縮放 | 60,000 |
   | `armor` | `armor-novice_helmet-stuff.png` | 只用這一張圖 → Hunyuan3D-2 單張圖模型 | 30,000 |
   | `appearance` | `appearance-hair_twin_tails-stuff.png` | 同上 | 40,000 |
   | `weapon` | `weapon-initiate_dagger-stuff.png` | 同上 | 20,000 |
   | `object` | `object-street_lamp-building.png`、`object-stone_a-building.png` | 同上（小型場景物件） | 30,000 |
   | `bigobject` | `bigobject-kronenberg_main_gate-front.png` ＋ `-back` | **正面、背面兩張都由你提供**（不生成視角）；**最高精度**：形狀網格 512、4K 貼圖（渲染 2048）、保留細小的裝飾零件 | 200,000 |

   `equipment`、`prop`、`environment` 也可以用，處理方式同 armor。結尾寫 `-front`、`-stuff` 或 `-building` 都可以。除了 bigobject（512），形狀解析度（octree）都是 384，物品只是面數較少。

   **monster、bigobject 一定要正面和背面兩張**，主圖結尾是 `-front`（bigobject 不能用 `-building`）：

   ```text
   monster-gelly-front.png                       bigobject-kronenberg_main_gate-front.png
   monster-gelly-back.png                        bigobject-kronenberg_main_gate-back.png
   ```

   缺背面時，區塊 4 會顯示 `WAITING … missing …-back` 並跳過，補上後（監看模式下會自動）再處理。兩類都不會用 AI 生成視角，側面由形狀模型推測；如果你另外放了 `-left`、`-right`，也會一起拿來建形狀（選用）。完成後所有用到的圖都改名為 `used-…`。

   **character 的額外圖（選用）**：可以另外放同 id 的 `-left`、`-back`、`-right` 圖，例如 `character-novice_female-back.png`。有放的角度直接採用，沒放的才用 MV-Adapter 生成；提供的 right 也會一起拿來建形狀（生成的 right 只供檢視）。方向規則：**left 圖裡主體的正面朝向畫面左邊，right 圖朝向右邊**（left 是面對正面時往你的右手邊繞 90 度看到的樣子，跟 Hunyuan3D-2mv 範例一致）。用 Gemini 生成時，可以要求「same building, viewed from directly behind / from its left side, same lighting, plain background」。完成後這些額外圖也會改名為 `used-…`。

   **bigobject** 是 T4 能做到的最高設定，每棟可能要 15～30 分鐘。形狀網格 512 或 4K 貼圖放不下時，會自動改用 384 網格或 2048 貼圖重試，不會讓整批失敗。貼圖細節仍受限於貼圖模型每個視角 512 像素的生成解析度，近看大型建築時，細部材質建議在 Godot 疊加可重複的材質貼圖。
3. 依序執行區塊 1～4。區塊 4 會處理 `in/` 裡所有未使用的正面圖，**每輪 3 張**（`claim_batch`）：每張圖貼完圖就立刻輸出到 `out/` 並把來源圖改名 `used-…`，不用等整批做完；中途斷線時，看 `out/` 和 `used-…` 就知道哪些已完成，其餘的圖重跑時從中斷的階段接著做。開著時繼續丟新圖也會接著做，閒置 `IDLE_STOP_MINUTES`（預設 30 分鐘）後自動停止。
4. 完成的檔案在 `out/`；區塊 5 可以看四視角預覽、列出失敗的圖、打包下載。

區塊 3 安裝完成後**不需要重新啟動**；新 runtime 第一次約 3～5 分鐘，之後同一個 runtime 幾秒完成。

### 正面圖怎麼準備（光頭模式）

RagnarokChange 的頭部和頭髮是另外的素材（04-D11），角色身體做成光頭最容易接上（怪物不受影響）。區塊 2 的 `HEAD_MODE` 預設 `bald`，生成視角時會要求光頭。**正面圖本身也是光頭時效果最好**，否則背面的頭髮可能時有時無。用 Gemini 生成正面圖時可以加上：

```text
same character and outfit, completely bald head with no hair, no helmet, no backpack, no weapons,
full body front view, A-pose, plain white background
```

背包、頭飾、武器另外做成單獨的素材（armor／weapon），不要畫在角色身上。透明背景的 PNG 最好；不透明的圖會自動去背（rembg `isnet-general-use`，設定在 `imageto3d/config.py` 的 `rembg_model`；舊的 u2net 會把灰色背景上的灰色劍刃整個去掉）。去背結果是 `-views.jpg` 的第一格，劍刃或淺色部分不見時就是去背的問題；要重做舊的圖，刪掉 `work/<類別>-<id>/` 並把 `used-` 圖改回原檔名。單張圖的物品會把圖裡看得到的東西都建出來，例如髮型圖裡的假人頭也會變成模型的一部分，所以髮型圖最好只有頭髮。

## Kaggle 版（2×T4，高精度）

檔名規則、輸出、Drive 資料夾都跟 Colab 版一樣，兩者可以輪流使用（**不要同時跑**，兩邊會搶 `in/` 的同一張圖）。

| 項目 | Colab 免費版 | Kaggle 版 |
|---|---|---|
| GPU | 1×T4 | 2×T4（各 16 GB，不是合成 32 GB）：每張卡各跑一份完整流程，從 `in/` 各拿 `CLAIM_BATCH` 張，不會重複 |
| 系統 RAM | 約 12.7 GB | 約 29 GB，比較不會被系統砍掉程序 |
| 精度 | 形狀網格 384、貼圖 2048（bigobject 例外） | `QUALITY = "high"`：所有類別形狀網格 512、4K 貼圖（渲染 2048）、較多面數（`imageto3d/config.py` 的 `HIGH_QUALITY_PRESETS`）；放不下時自動降回 384／2048 |
| Drive | 直接掛載 | 沒有掛載功能。`DRIVE_MODE = "manual"`（預設）：圖從 Drive 分享連結或 Kaggle Dataset 拿，成果打包成 zip 從 Output 下載；`"rclone"`：同步到相同的本機路徑 `/content/drive/MyDrive/RagnarokChange/hunyuan3d` |
| 模型 | Drive 的 `hf_cache/` | 你自己的 private Kaggle Dataset，掛在 `/kaggle/input/…`，不用下載 |
| 執行 | 要開著瀏覽器 | Save & Run All 在背景跑，最多 12 小時；`STOP_AFTER_HOURS`（預設 11）後不再開始新的圖，把成果傳回 Drive 才結束 |

`VIEW_COUNT` 仍預設 4：6 視角多出的 45°、315° 不會用在建形狀，只多花顯存和時間。貼圖模型每個視角固定 512 px，這是模型本身的上限，GPU 多也無法提高。

### 一次性設定

**1. Kaggle 帳號**：到 kaggle.com/settings 做手機驗證（才能用 GPU 和網路）。

**2.（只有 `DRIVE_MODE = "rclone"` 需要）讓 Kaggle 讀寫 Google Drive**：在自己的電腦安裝 [rclone](https://rclone.org/downloads/)，然後：

```bash
rclone config
#   n                    新增 remote
#   name> gdrive         名稱要跟 notebook 的 REMOTE 一致（gdrive:RagnarokChange/hunyuan3d）
#   Storage> drive       Google Drive
#   client_id、client_secret 直接按 Enter（新版 rclone 不能留空時，要先在 Google Cloud Console 建立
#   「電腦版應用程式」OAuth 用戶端並把應用程式發布為正式版，否則 token 7 天後失效）
#   scope> 1             完整存取（要能改名 in/ 的圖、寫 out/）
#   service_account_file 按 Enter；Edit advanced config? n；Use web browser? y（在瀏覽器登入並同意）
#   Configure as Shared Drive? n → y 確認 → q 離開
rclone lsd gdrive:RagnarokChange/hunyuan3d   # 看得到 in、out、work 就成功
rclone config file                            # 顯示 rclone.conf 的位置
```

在 Kaggle 任一 notebook 的 **Add-ons → Secrets** 新增 `RCLONE_CONF`，值貼上 rclone.conf 的**全部內容**。這個 token 可以存取你整個 Drive：只放在 Secrets，notebook 保持 Private。

**3. Kaggle API token**（上傳模型 Dataset 用）：kaggle.com/settings → API → **Create New Token**。新版 token 存成 Secret `KAGGLE_API_TOKEN`；舊版 `kaggle.json` 則把裡面的 `key` 存成 `KAGGLE_KEY`。

**4. 選用**：Hugging Face token 存成 Secret `HF_TOKEN`（下載限速或遇到 401/403 時）。

### 建立模型 Dataset（只做一次，用 CPU）

1. Kaggle → Create → **Import Notebook**，匯入 [`kaggle_prepare_models.ipynb`](kaggle_prepare_models.ipynb)。
2. Session options：Accelerator **None**、Internet **on**；Secrets 勾選 `KAGGLE_API_TOKEN`（或 `KAGGLE_KEY`）、`HF_TOKEN`。
3. Run All（或 Save & Run All 在背景跑）。不含 turbo 約 25～35 GB；Kaggle 機房連 Hugging Face 通常很快，每個 repo 的速度會印出來。
4. 完成後打開印出的 Dataset 網址，確認是 **Private**，檔案裡看得到 `tencent/Hunyuan3D-2`、`imageto3d-models.json` 等。

這個 notebook 下載的內容等同於下面的指令（`huggingface_hub` 0.36 的 `hf` 指令），只下載流程用得到的檔案，同時下載去背模型 `rembg/u2net.onnx`：

```bash
hf download stabilityai/stable-diffusion-xl-base-1.0 --local-dir models/stabilityai/stable-diffusion-xl-base-1.0 \
  --include "model_index.json" "scheduler/*" "tokenizer/*" "tokenizer_2/*" \
            "text_encoder/config.json" "text_encoder/model.fp16.safetensors" \
            "text_encoder_2/config.json" "text_encoder_2/model.fp16.safetensors" \
            "unet/config.json" "unet/diffusion_pytorch_model.fp16.safetensors" \
            "vae/config.json" "vae/diffusion_pytorch_model.fp16.safetensors"
hf download madebyollin/sdxl-vae-fp16-fix --local-dir models/madebyollin/sdxl-vae-fp16-fix \
  --include "config.json" "diffusion_pytorch_model.safetensors"
hf download huanngzh/mv-adapter --local-dir models/huanngzh/mv-adapter --include "mvadapter_i2mv_sdxl.safetensors"
hf download tencent/Hunyuan3D-2mv --local-dir models/tencent/Hunyuan3D-2mv \
  --include "hunyuan3d-dit-v2-mv/config.yaml" "hunyuan3d-dit-v2-mv/model.fp16.safetensors"
hf download tencent/Hunyuan3D-2 --local-dir models/tencent/Hunyuan3D-2 \
  --include "hunyuan3d-dit-v2-0/config.yaml" "hunyuan3d-dit-v2-0/model.fp16.safetensors" \
            "hunyuan3d-delight-v2-0/*" "hunyuan3d-paint-v2-0/*"
```

要用 turbo 模型時，把 `INCLUDE_TURBO = True` 再跑一次（會成為同一個 Dataset 的新版本）。Dataset 裡沒有的模型會在生成時自動從 Hugging Face 下載到 session 的暫存空間。

### 每次使用

1. 匯入 [`hunyuan3d2_kaggle_2xt4.ipynb`](hunyuan3d2_kaggle_2xt4.ipynb)（第一次之後直接開啟）。
2. Session options：Accelerator **GPU T4 x2**、Internet **on**；**Add Input** → Your Work → Datasets 加入模型 Dataset。
3. 準備圖（檔名規則同上），**manual 模式**二選一，也可以同時用：
   - **Drive 分享連結**：在 Drive 建一個只放這次新圖的資料夾（例如 `kaggle_in`，**最多 50 個檔案**），共用 →「知道連結的任何人」可檢視，把連結貼到區塊 2 的 `IN_FOLDER_URL`。
   - **Kaggle Dataset**：kaggle.com → Create → New Dataset，把圖拖進去（Private），再用 Add Input 加入。之後新增圖就在 Dataset 頁面 New Version。
4. 想看過程就依序執行區塊 1～4；要整晚跑就按 **Save Version → Save & Run All (Commit)**，然後可以關掉瀏覽器。
5. 每張圖做完就立刻寫出 GLB 到 `/kaggle/working/hunyuan3d/out/`，再把 Kaggle 這邊 `in/` 裡它用到的圖改名 `used-…`，然後才做下一張。圖只在區塊 2 讀一次，全部做完就結束；想在跑的過程中繼續丟圖，設 `RECHECK_FOLDER = True`，會每 `SYNC_MINUTES`（預設 5 分鐘）再檢查分享資料夾（Dataset 在 session 中不會變）。
6. 全部做完（或到 `STOP_AFTER_HOURS`）後，`in/` 和 `out/` 打包成 `/kaggle/working/hunyuan3d-results-<時間>.zip`：互動執行時在右側 **Output** 面板下載；背景執行時到該版本頁面的 **Output** 分頁下載。中途斷掉沒打包時，`/kaggle/working/hunyuan3d/` 也可以直接下載（互動 session 還開著時）。`out/` 的檔案放進 Drive 的 `out/`。
7. **Kaggle 不能改你 Drive 裡的檔名**：`used-…` 只出現在 Kaggle 這邊的 `in/`（zip 裡的 `in/`）。照著 zip 的 `in/`，把 Drive 分享資料夾裡對應的圖也改名 `used-…` 或移走，不然下次 session 會再做一次；沒有 `used-` 的就是還沒做完。

manual 模式下，沒做完的圖下次 session 要從頭做（中間檔不會保留），CUDA 擴充每次 session 也要重新編譯（約數分鐘）。`DRIVE_MODE = "rclone"` 時則會：每 `SYNC_MINUTES`（預設 5 分鐘）把成品傳回 Drive 的 `out/`、完成的圖在 Drive 改名 `used-…`、中間檔傳回 `work/` 讓下次 session 接著做、編譯好的擴充存到 Drive 的 `wheels/`。

### 時間與免費額度（估計，以實際的 log 為準）

每張圖在單張 T4、`QUALITY = "high"` 下的估計時間：character 15～20 分鐘、monster 12～15 分鐘、armor／weapon／appearance／object 8～12 分鐘、bigobject 25～40 分鐘。log 的 `[stage] <id>: done in … s` 是實際秒數。

Kaggle GPU 免費額度每週約 30 小時（以 kaggle.com 帳號頁顯示為準），每個 session 最多 12 小時。兩張卡同時做，每週約可完成：全部 character 約 170～220 個、全部小物件約 280～400 個、全部 bigobject 約 85～130 棟；混合約 150～250 個。

## Kaggle 文字生圖（prompts.json → 透明 PNG）

[`kaggle_imagegen_2xt4.ipynb`](kaggle_imagegen_2xt4.ipynb) 依 `prompts.json` 產生可直接放進 `in/` 的圖，同樣用 2×T4、背景執行、跑完打包 zip。

- **正面**：FLUX.1-schnell（Apache-2.0；4-bit 量化放進 T4，T5 文字編碼用 bfloat16 算完就卸載）。描述後面自動加上：類別提示（character：A-pose、光頭、不拿武器；weapon：直立、尖端朝上；appearance：只有頭髮…）、寫實風格（`photorealistic high-detail game asset, realistic PBR materials`）、單一物體置中完整入鏡、中灰背景、平光無陰影。中灰背景是為了避免淺色物體在去背時被當成背景。
- **其他視角**：以生成的正面為準用 MV-Adapter 生成，所以各視角是同一個物體（畫質略低於正面，768 px）。
- **去背**：isnet，存成透明 PNG，3D 流程不會再去背一次。
- **兩張 T4**：各做一半的項目；先做所有正面，再做其他視角（分開的程序，記憶體不會疊加）。
- **跳過做完的**：zip 裡的 `prompts.json` 把做完的項目標成 `"done": true`；用它取代 Drive 上的那份，下次只做沒完成的。
- 失敗的項目不影響其他項目，原因印在區塊 4 最後。

```json
{
  "defaults": {"seed": 1000, "style": "photorealistic high-detail game asset, realistic PBR materials"},
  "items": [
    {"name": "weapon-two_hand_brokensword", "prompt": "a chipped steel two-handed greatsword, leather grip, round pommel"},
    {"name": "monster-gelly", "prompt": "a small green jelly slime monster with big eyes", "views": ["front", "back"]},
    {"name": "bigobject-kronenberg_main_gate", "prompt": "a medieval stone city gate with two towers", "views": ["front", "back"], "seed": 7}
  ]
}
```

| 欄位 | 說明 |
|---|---|
| `name` | `<類別>-<id>`，類別同 3D 流程；輸出 `<name>-<view>.png` |
| `prompt` | 物體本身的英文描述；背景、構圖、風格會自動加上 |
| `views` | `front`、`left`、`back`、`right` 任選；省略時只做正面；正面一定會做 |
| `seed`、`style` | 選用；`defaults` 裡的值套用到全部項目，項目自己的優先 |
| `done` | 跑完自動加上；拿掉就會重做 |

monster、bigobject 記得要 `"views": ["front", "back"]`（3D 流程需要背面）。character 只做正面也可以，3D 流程會自己補視角。

**時間（估計）**：FLUX 每張正面約 30～60 秒，補視角每項約 1～2 分鐘；第一次 session 要下載 FLUX（約 34 GB）。

## 流程與記憶體

```text
in/<類別>-<id>-<front|stuff>.png
  │ 1. views   去背；monster／bigobject 只用你提供的圖（正面＋背面，可加左右）；character 缺的視角用 MV-Adapter SDXL 生成
  │ 2. shape   多視角類別：Hunyuan3D-2mv 用正、左、背（有提供 right 時加上 right）；其他：Hunyuan3D-2 只用正面 → 移除碎片 → 減面
  │ 3. paint   Hunyuan3D-2 依正面圖上貼圖
  │ 4. export  縮放到真實尺寸、腳踩地面、PBR 材質 → out/，正面圖改名 used-…
out/hunyuan3d-<類別>-<id>.glb / .json / -views.jpg
```

- **每個階段是獨立的 Python 程序**：一次載入一個模型、處理完這一輪的圖（`claim_batch` 張，預設 3）後結束，結束時 RAM 和 GPU 記憶體全部還給系統，所以連續跑很多張也不會累積到耗盡。多視角和單張圖要用不同的形狀模型，所以這兩組各自用一個程序。
- **各模型都改成省記憶體的載入方式**：SDXL 以 fp16 載入、adapter 與形狀模型直接在 GPU 上以 fp16 建立、貼圖模型的權重用 mmap 按需讀取，避免免費版 12.7 GB 系統 RAM 爆掉。
- **失敗隔離**：某張圖出錯（包括記憶體不足被系統終止）只記在那張圖上，下一輪再試一次，連續失敗兩次就跳過，其他圖照常處理。錯誤記在 `work/<類別>-<id>/status.json`。
- **可以中斷**：中間檔存在 Drive 的 `work/`，斷線後重跑區塊 1～4，會從中斷的階段接著做。

## 輸出（給 RagnarokChange 用）

| 檔案 | 內容 |
|---|---|
| `hunyuan3d-<類別>-<id>.glb` | 網格與基本色貼圖（PBR 材質，金屬度 0、粗糙度 0.9）；1 單位 = 1 公尺、+Y 朝上、面向 +Z；角色縮放到身高 1.65 m（id 含 `male` 而不含 `female` 時 1.78 m），其他類別保留原尺寸；腳踩在 y = 0 |
| `hunyuan3d-<類別>-<id>.json` | 授權代碼、來源圖、尺寸、面數、所有模型與種子，可用來填 `assets/license_registry.json` |
| `hunyuan3d-<類別>-<id>-views.jpg` | 正、左、背、右四視角預覽 |

檔名照 RagnarokChange `docs/04-art-pipeline.md` 第 6.2 節的 `<來源>-<類別>-<名稱>` 格式，類別沿用你的檔名用字（armor、weapon 對應那邊的 equipment，appearance 對應 characters 的髮型）。物品類保留生成時的尺寸（約 2 單位高），實際尺寸由那邊的 Blender 管線依概念圖標示縮放。法線、ORM 貼圖、LOD、綁骨由 RagnarokChange 的 Blender 管線與 Mixamo 處理。

**授權**：JSON 裡的授權代碼是 `LicenseRef-Tencent-Hunyuan3D-Community`。Hunyuan3D 不在 RagnarokChange 04-D03 的白名單上，使用前要在那邊的文件決定是否加入。

## 雲端硬碟與資料夾

```text
Google Drive / MyDrive / RagnarokChange / hunyuan3d /
├── in/            # 放正面圖；完成後改名為 used-…
├── out/           # 完成的 GLB、JSON、四視角預覽
├── work/          # 每張圖的中間檔：front.png、left/back/right.jpg、shape.glb、textured.glb、status.json
├── hf_cache/      # Hugging Face 模型權重（約 42 GB，只下載一次）
├── rembg_cache/   # 去背模型
└── wheels/        # 編譯好的 CUDA 擴充（依 Python/torch/GPU 分資料夾）
```

資料夾會在區塊 2 自動建立。想換位置就改區塊 2 的 `DRIVE_ROOT`。

## 可修改設定（區塊 2）

| 設定 | 預設 | 說明 |
|---|---|---|
| `HEAD_MODE` | `bald` | 只影響 character：`bald` 生成光頭；`keep` 照正面圖 |
| `VIEW_COUNT` | 4 | 4 視角較省顯存；6 是 MV-Adapter 官方設定，約需 14 GB VRAM |
| `VIEW_SEED` | 1234 | 視角不理想時換一個數字 |
| `SHAPE_MODEL` | 完整版 | character／monster／bigobject 的形狀模型（多視角）。完整版細節較好（T4 每個形狀約 3 分鐘）；turbo 約 30 秒 |
| `SINGLE_SHAPE_MODEL` | 完整版 | armor／appearance／weapon 的形狀模型（Hunyuan3D-2 單張圖） |
| `PAINT_MODEL` | 完整版 | turbo 較快 |
| `TEXTURE_SIZE` | 2048 | 顯存不足時改 1024 |
| `claim_batch`（`imageto3d/config.py`） | 3 | 每輪幾張圖。每輪每個模型只載入一次；數字小，完成的圖出來得快，數字大，載入模型的次數少 |
| `WATCH`、`POLL_MINUTES`、`IDLE_STOP_MINUTES` | 開、2、30 | 持續監看 `in/` 的頻率與閒置多久停止 |
| `PIPELINE_REF` | 目前分支 | 流程程式碼的版本；合併到 main 後改成 `main` |

各類別的面數上限與形狀解析度在 `imageto3d/config.py` 的 `presets`。

## 程式結構

| 檔案 | 用途 |
|---|---|
| `hunyuan3d2_colab_t4.ipynb` | Colab 介面：檢查環境、設定、安裝、執行批次、看結果 |
| `hunyuan3d2_kaggle_2xt4.ipynb` | Kaggle 介面：rclone 同步 Drive、每張 GPU 一個 worker、高精度設定 |
| `kaggle_prepare_models.ipynb` | Kaggle（CPU）：下載模型、上傳成 private Dataset |
| `kaggle_imagegen_2xt4.ipynb` | Kaggle：prompts.json → FLUX 正面 ＋ MV-Adapter 其他視角 → 透明 PNG |
| `imageto3d/run.py` | 批次主程式：掃描 `in/`，依序啟動各階段程序，監看新檔案；`--worker N` 時跟其他 GPU 分工 |
| `imageto3d/drive_sync.py` | Kaggle rclone 模式：同步 Drive 的 `in/`、`out/`、`work/`、`wheels/` |
| `imageto3d/kaggle_io.py` | Kaggle manual 模式：從分享資料夾和 Dataset 收集圖、打包成果 |
| `imageto3d/imagegen.py` | 文字生圖：讀 prompts.json、組合描述、FLUX 與 MV-Adapter 階段、標記完成 |
| `imageto3d/models.py` | 模型清單、下載到 Dataset 資料夾、在 `/kaggle/input` 找 Dataset |
| `imageto3d/stage_views.py`、`stage_shape.py`、`stage_paint.py`、`stage_export.py` | 四個階段，各自是獨立程序 |
| `imageto3d/assets.py` | 檔名解析、各階段是否需要重做、失敗紀錄、多 GPU 時的認領檔 |
| `imageto3d/runtime.py` | 記憶體相關工具（fp16 直接建在 GPU、mmap 讀權重、釋放記憶體） |
| `imageto3d/config.py` | 所有設定與預設值 |

模型輸出受輸入圖、種子和模型限制；生成的背面不保證完全符合概念圖。各模型及權重受其各自授權條款限制（MV-Adapter、SDXL、Tencent Hunyuan3D-2 Community License）。
