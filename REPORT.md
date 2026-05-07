# Lab 21 — Evaluation Report: QLoRA Router Agent cho Chatbot Tuyển sinh

**Học viên**: Thái Doãn Minh Hải
**Ngày nộp**: 2026-05-07  
**Submission option**: B — GitHub/HuggingFace Hub style submission  
**HF Hub adapter**: https://huggingface.co/minhhai0227/admissions-router-qwen2p5-3b-r16

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit với Unsloth + PEFT + TRL `SFTTrainer`
- **Task**: fine-tune router agent cho chatbot tuyển sinh VinUniversity.
- **Dataset**: `admissions_router_dataset_v1.jsonl`, gồm **210 samples** dạng Alpaca `instruction / input / output`.
- **Train/eval split**: 90/10, tương đương khoảng **189 train** và **21 eval** samples.
- **Dataset domain**: câu hỏi tuyển sinh, gồm route cho học phí, học bổng, quy trình/deadline, điều kiện tuyển sinh, thông tin chương trình, thông tin trường và câu hỏi general.
- **Intent distribution**: `tuition_lookup` 55, `program_info` 53, `scholarship_lookup` 36, `admission_requirement` 35, `general_question` 16, `timeline_process` 9, `school_info` 6.
- **Answer mode distribution**: `retrieve` 184, `direct` 16, `clarify` 10.
- **Token length analysis**: p95 = **288 tokens**, `max_seq_length` chọn = **512**.
- **LoRA config baseline**: `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj"]`, `lora_dropout=0`, gradient checkpointing bật.
- **Training hyperparameters**: 3 epochs, learning rate `2e-4`, cosine scheduler, warmup ratio `0.10`, effective batch size = 8, optimizer `adamw_8bit`.
- **GPU**: Google Colab T4, khoảng 16GB VRAM.
- **Training cost ước tính**: $0 nếu dùng Colab Free; tổng thời gian train 3 rank ghi nhận khoảng **11.89 phút**, chưa tính thời gian install/eval/upload.

Ảnh phân bố token được lưu tại: `results/token_length_distribution.png`.

## 2. Rank Experiment Results

Bảng dưới đây so sánh 3 rank LoRA trên cùng dataset, cùng model và cùng hyperparameters, chỉ thay `rank` và `lora_alpha`.

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 3.86 min | 4.25 GB | 0.1939 | 1.2140 |
| 16 | 32 | 3,686,400 | 4.14 min | 3.65 GB | 0.1524 | 1.1646 |
| 64 | 128 | 14,745,600 | 3.89 min | 5.02 GB | 0.0739 | 1.0767 |


**Nhận xét chính**:

- `r=64` có eval loss/perplexity tốt nhất: perplexity **1.0767**, thấp hơn `r=16` là **1.1646** và `r=8` là **1.2140**.
- `r=64` có số trainable params **14,745,600**, gấp 4 lần `r=16` và 8 lần `r=8`, nên adapter nặng hơn và tốn tài nguyên hơn.
- Peak VRAM của `r=64` là **5.02GB**, cao nhất trong 3 cấu hình. `r=16` có VRAM thấp nhất trong log (**3.65GB**), có thể do biến động đo peak memory sau cleanup/caching trên Colab.
- Training time giữa các rank khá gần nhau trong lần chạy này. `r=64` hơi nhanh hơn `r=16` trong CSV, nhưng sự khác biệt nhỏ này nhiều khả năng do variance của Colab/caching, không nên kết luận `r=64` luôn nhanh hơn.

## 3. Loss Curve Analysis

Loss curve baseline `r=16` được lưu tại: `results/loss_curve_r16.png`.

Quan sát từ biểu đồ:

- Training loss giảm mạnh từ khoảng **2.3** xuống dưới **0.2** sau khoảng 40–50 steps.
- Đường loss sau đó đi ngang quanh vùng thấp, cho thấy model đã học được pattern output JSON của router agent.
- Eval point cuối khoảng **0.1524**, tương ứng perplexity **1.1646**.
- Không thấy dấu hiệu overfitting rõ ràng trong run này vì eval loss vẫn thấp và không có xu hướng tăng so với train loss cuối. Tuy nhiên, dataset chỉ có 210 samples và phần eval nhỏ, nên kết luận overfitting cần thận trọng. Với production, nên tăng eval set bằng log user thật và thêm nhiều case khó/ngoài phân phối.

Một điểm cần lưu ý là task router có output rất ngắn và schema cố định, nên loss có thể giảm nhanh hơn so với task sinh câu trả lời dài. Điều này phù hợp với mục tiêu fine-tune router thay vì fine-tune toàn bộ chatbot trả lời factual knowledge.

## 4. Qualitative Comparison

Bảng dưới đây lấy 6 ví dụ đầu từ `qualitative_comparison.csv`. Trường `finetuned_json_valid=True` cho thấy output của adapter r16 parse được thành JSON hợp lệ.

| # | Prompt | Base output | Fine-tuned r16 output | Nhận xét |
|---:|---|---|---|---|
| 1 | Ngành Khoa học Máy tính học phí bao nhiêu? | `{"intent":"tuition_lookup","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"học phí ngành Khoa học Máy tính VinUni"}` | `{"intent":"tuition_lookup","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"học phí ngành Khoa học Máy tính VinUni"}` | JSON hợp lệ; output r16 khớp schema. Trong CSV lần chạy này, base và r16 giống nhau. |
| 2 | Học phí bao nhiêu? | `{"intent":"tuition_lookup","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"học phí VinUni"}` | `{"intent":"tuition_lookup","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"học phí VinUni"}` | JSON hợp lệ; output r16 khớp schema. Trong CSV lần chạy này, base và r16 giống nhau. |
| 3 | Em muốn biết học bổng VinUni cần điều kiện gì? | `{"intent":"scholarship_lookup","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"học bổng VinUni cần điều kiện gì"}` | `{"intent":"scholarship_lookup","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"học bổng VinUni cần điều kiện gì"}` | JSON hợp lệ; output r16 khớp schema. Trong CSV lần chạy này, base và r16 giống nhau. |
| 4 | Deadline nộp hồ sơ năm 2026 là khi nào? | `{"intent":"timeline_process","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"thời gian nộp hồ sơ VinUni 2026"}` | `{"intent":"timeline_process","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"thời gian nộp hồ sơ VinUni 2026"}` | JSON hợp lệ; output r16 khớp schema. Trong CSV lần chạy này, base và r16 giống nhau. |
| 5 | Ngành Data Science học những gì? | `{"intent":"program_info","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"Thông tin chương trình học Data Science VinUni"}` | `{"intent":"program_info","answer_mode":"retrieve","needs_retrieval":true,"needs_clarification":false,"clarification_question":null,"transformed_query":"Thông tin chương trình học Data Science VinUni"}` | JSON hợp lệ; output r16 khớp schema. Trong CSV lần chạy này, base và r16 giống nhau. |
| 6 | Cảm ơn bạn nhé | `{"intent":"general_question","answer_mode":"direct","needs_retrieval":false,"needs_clarification":false,"clarification_question":null,"transformed_query":null}` | `{"intent":"general_question","answer_mode":"direct","needs_retrieval":false,"needs_clarification":false,"clarification_question":null,"transformed_query":null}` | JSON hợp lệ; output r16 khớp schema. Trong CSV lần chạy này, base và r16 giống nhau. |


**Đánh giá qualitative**:

- 10/10 qualitative prompts trong CSV có output r16 parse được JSON hợp lệ.
- Các câu như hỏi học phí, học bổng, deadline, thông tin ngành và cảm ơn đều được route theo schema mong muốn.
- Trong file qualitative của lần chạy này, `base_output` và `finetuned_r16_output` đang giống nhau ở tất cả ví dụ. Vì vậy phần qualitative hiện kiểm chứng tốt về format JSON, nhưng chưa phản ánh rõ before/after gap giữa base và fine-tuned model. Lần chạy sau nên reload một base model sạch để generate base output trước, sau đó mới load adapter r16 để so sánh công bằng hơn.
- Có một case cần cải thiện trong qualitative: câu “Có cần IELTS để vào ngành AI không?” sinh intent `scholarship_requirement`, trong khi allowed intents không có intent này. Điều này cho thấy cần thêm schema validator hoặc train thêm negative/edge cases để ép output nằm trong enum hợp lệ.

Ngoài qualitative comparison, evaluation router trên 21 mẫu eval cho kết quả:

| Metric | Result |
|---|---:|
| JSON valid rate | 90.48% |
| Intent accuracy | 85.71% |
| Answer mode accuracy | 85.71% |
| Needs retrieval accuracy | 90.48% |
| Needs clarification accuracy | 85.71% |

Các lỗi chính trong `router_eval_details.csv` gồm: một số output lặp từ “VinUni” quá nhiều nên không parse được JSON hoàn chỉnh; một số câu general như “Tôi đang học lớp 12” bị chuyển sang `clarify`; và một số câu về chương trình/viện có nhầm giữa `program_info` và `school_info`.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, `r=64` đạt perplexity tốt nhất, nhưng với product chatbot tuyển sinh, rank tốt nhất để bắt đầu deploy nên là **r=16**. Lý do là task router agent không yêu cầu sinh văn bản dài hay học kiến thức phức tạp; nó chủ yếu học mapping từ câu hỏi user sang JSON gồm `intent`, `answer_mode`, `needs_retrieval`, `needs_clarification` và `transformed_query`. Vì output ngắn và schema cố định, `r=16` đã cho perplexity thấp (**1.1646**) với số trainable params chỉ **3.69M**, nhẹ hơn nhiều so với `r=64` (**14.75M**). `r=8` cũng là lựa chọn đáng cân nhắc nếu ưu tiên chi phí và tốc độ, nhưng perplexity cao hơn `r=16`. `r=64` có chất lượng định lượng tốt nhất, nhưng lợi ích tăng thêm có thể không đáng nếu xét production deployment, đặc biệt khi cần model server nhanh, nhẹ và dễ update. Recommendation của tôi là dùng **r=16** làm baseline an toàn, sau đó thu thêm log user thật, mở rộng dataset clarify/out-of-scope, rồi thử lại `r=8` để xem có thể giảm chi phí inference/training mà vẫn giữ accuracy đủ tốt hay không.

## 6. What I Learned

- Fine-tuning không nên dùng để nhét kiến thức tuyển sinh như học phí, deadline hay học bổng mới nhất. Những thông tin này vẫn phải để RAG/Qdrant xử lý. LoRA/QLoRA phù hợp hơn để dạy model **hành vi**, ví dụ route câu hỏi, giữ JSON schema và chọn `direct/clarify/retrieve`.
- Rank LoRA là một trade-off thực tế. Rank cao hơn có thể giảm eval loss/perplexity, nhưng tăng trainable params và VRAM. Với task router ngắn, `r=16` thường hợp lý hơn `r=64` nếu xét ROI.
- Dataset quan trọng hơn model. Các lỗi như output lặp “VinUni”, intent ngoài enum, hoặc nhầm giữa `program_info` và `school_info` cho thấy cần review dataset kỹ hơn, thêm schema validator và thu nhiều log user thật trước khi dùng trong production.

## Appendix — Files

- `results/rank_experiment_summary.csv`: bảng so sánh rank r=8, r=16, r=64.
- `results/qualitative_comparison.csv`: so sánh qualitative base vs fine-tuned r16.
- `results/router_eval_metrics.json`: accuracy/valid JSON metrics.
- `results/router_eval_details.csv`: chi tiết từng eval sample.
- `results/loss_curve_r16.png`: loss curve baseline r16.
- `results/token_length_distribution.png`: token length analysis.
- `data/admissions_router_dataset_v1.jsonl`: dataset router agent dùng cho lab.
