// dev/creator=tubakhxn

#include <opencv2/opencv.hpp>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/port/file_helpers.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::string kBaseDir = ".";
const std::string kOutputDir = kBaseDir + "/output";
const std::string kScreenshotDir = kOutputDir + "/screenshots";
const std::string kRecordingDir = kOutputDir + "/recordings";
const std::string kLogDir = kOutputDir + "/logs";

const std::string kGraphConfig = R"pb(
input_stream: "input_video"
output_stream: "landmarks"
output_stream: "handedness"
node {
  calculator: "HandLandmarkTrackingCpu"
  input_stream: "IMAGE:input_video"
  output_stream: "LANDMARKS:landmarks"
  output_stream: "HANDEDNESS:handedness"
  node_options: {
    [type.googleapis.com/mediapipe.HandLandmarkTrackingCpuOptions] {
      num_hands: 1
      min_detection_confidence: 0.7
      min_tracking_confidence: 0.6
    }
  }
}
)pb";

struct Landmark3D {
    float x, y, z;
};

struct FingerExtension {
    float thumb, index, middle, ring, pinky;
};

struct HandState {
    FingerExtension extension;
    bool thumb_across_palm;
    float spread_index_middle;
    float thumb_index_tip_distance;
    float palm_size;
};

static float Distance(const Landmark3D &a, const Landmark3D &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static float JointAngleDegrees(const Landmark3D &a, const Landmark3D &b, const Landmark3D &c) {
    float v1x = a.x - b.x, v1y = a.y - b.y, v1z = a.z - b.z;
    float v2x = c.x - b.x, v2y = c.y - b.y, v2z = c.z - b.z;
    float n1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z) + 1e-6f;
    float n2 = std::sqrt(v2x * v2x + v2y * v2y + v2z * v2z) + 1e-6f;
    float dot = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2);
    dot = std::max(-1.0f, std::min(1.0f, dot));
    return std::acos(dot) * 180.0f / static_cast<float>(M_PI);
}

static float ExtensionFromAngle(float angleDeg) {
    if (angleDeg > 150.0f) return 1.0f;
    if (angleDeg < 100.0f) return 0.0f;
    return (angleDeg - 100.0f) / 50.0f;
}

static HandState ComputeHandState(const std::vector<Landmark3D> &lm) {
    HandState state{};
    const Landmark3D &wrist = lm[0];
    const Landmark3D &middleMcp = lm[9];
    state.palm_size = Distance(middleMcp, wrist) + 1e-6f;

    state.extension.thumb = ExtensionFromAngle(JointAngleDegrees(lm[2], lm[3], lm[4]));
    state.extension.index = ExtensionFromAngle(JointAngleDegrees(lm[5], lm[6], lm[8]));
    state.extension.middle = ExtensionFromAngle(JointAngleDegrees(lm[9], lm[10], lm[12]));
    state.extension.ring = ExtensionFromAngle(JointAngleDegrees(lm[13], lm[14], lm[16]));
    state.extension.pinky = ExtensionFromAngle(JointAngleDegrees(lm[17], lm[18], lm[20]));

    float thumbToIndexMcp = Distance(lm[4], lm[5]) / state.palm_size;
    state.thumb_across_palm = thumbToIndexMcp < 0.55f;

    state.spread_index_middle = Distance(lm[8], lm[12]) / state.palm_size;
    state.thumb_index_tip_distance = Distance(lm[4], lm[8]) / state.palm_size;

    return state;
}

struct Prediction {
    std::string label;
    float confidence;
};

static Prediction ClassifyGesture(const std::vector<Landmark3D> &lm) {
    HandState s = ComputeHandState(lm);
    const auto &ext = s.extension;

    bool thumbExt = ext.thumb > 0.55f;
    bool indexExt = ext.index > 0.55f;
    bool middleExt = ext.middle > 0.55f;
    bool ringExt = ext.ring > 0.55f;
    bool pinkyExt = ext.pinky > 0.55f;

    bool indexCurl = ext.index < 0.45f;
    bool middleCurl = ext.middle < 0.45f;
    bool ringCurl = ext.ring < 0.45f;
    bool pinkyCurl = ext.pinky < 0.45f;
    bool thumbCurl = ext.thumb < 0.45f;

    int extendedCount = (indexExt ? 1 : 0) + (middleExt ? 1 : 0) + (ringExt ? 1 : 0) + (pinkyExt ? 1 : 0);
    int curledCount = (indexCurl ? 1 : 0) + (middleCurl ? 1 : 0) + (ringCurl ? 1 : 0) + (pinkyCurl ? 1 : 0);

    bool pinch = s.thumb_index_tip_distance < 0.35f;
    bool thumbOverFingers = s.thumb_across_palm;

    std::vector<Prediction> candidates;

    if (indexExt && middleCurl && ringCurl && pinkyCurl && !thumbExt)
        candidates.push_back({"D", 0.92f});
    if (indexExt && middleCurl && ringCurl && pinkyCurl && thumbExt)
        candidates.push_back({"L", 0.90f});
    if (indexExt && middleExt && ringCurl && pinkyCurl) {
        float gap = s.spread_index_middle;
        if (gap > 0.32f) candidates.push_back({"V", 0.88f});
        else if (thumbExt) candidates.push_back({"K", 0.80f});
        else if (gap < 0.15f) candidates.push_back({"R", 0.75f});
        else candidates.push_back({"U", 0.85f});
    }
    if (indexExt && middleExt && ringExt && !pinkyExt)
        candidates.push_back({"W", 0.88f});
    if (extendedCount == 4 && !thumbExt)
        candidates.push_back({"B", 0.92f});
    if (curledCount == 4) {
        if (thumbOverFingers && !thumbExt) candidates.push_back({"S", 0.88f});
        else if (thumbExt && !thumbOverFingers) candidates.push_back({"A", 0.88f});
        else if (thumbExt && thumbOverFingers && pinch) candidates.push_back({"T", 0.70f});
        else if (thumbCurl && thumbOverFingers) candidates.push_back({"N", 0.55f});
        else if (thumbCurl) candidates.push_back({"M", 0.50f});
    }
    if (pinkyExt && thumbExt && indexCurl && middleCurl && ringCurl)
        candidates.push_back({"Y", 0.90f});
    if (indexExt && pinkyExt && !middleExt && !ringExt && !thumbExt)
        candidates.push_back({"H", 0.55f});
    if (!indexExt && pinkyExt && middleCurl && ringCurl && thumbCurl)
        candidates.push_back({"I", 0.85f});
    if (pinch && middleExt && ringExt && pinkyExt)
        candidates.push_back({"F", 0.85f});
    if (pinch && middleCurl && ringCurl && pinkyCurl)
        candidates.push_back({"O", 0.80f});
    if (extendedCount == 0 && !thumbExt && !pinch)
        candidates.push_back({"C", 0.55f});
    if (extendedCount == 0 && thumbExt && !thumbOverFingers)
        candidates.push_back({"E", 0.50f});
    if (indexExt && middleCurl && ringCurl && pinkyCurl && thumbExt && s.spread_index_middle < 0.2f)
        candidates.push_back({"X", 0.50f});
    if (extendedCount == 4 && thumbExt)
        candidates.push_back({"NOTHING", 0.40f});

    if (candidates.empty()) return {"UNKNOWN", 0.25f};

    Prediction best = candidates[0];
    for (const auto &c : candidates) {
        if (c.confidence > best.confidence) best = c;
    }
    return best;
}

class PredictionSmoother {
public:
    PredictionSmoother(int bufferSize = 12, float confidenceThreshold = 0.5f, int stabilityFrames = 8)
        : bufferSize_(bufferSize), confidenceThreshold_(confidenceThreshold), stabilityFrames_(stabilityFrames) {}

    std::string Update(const std::string &label, float confidence, float &stabilityRatio, bool &committed) {
        std::string effectiveLabel = confidence < confidenceThreshold_ ? "UNKNOWN" : label;
        buffer_.push_back(effectiveLabel);
        if (static_cast<int>(buffer_.size()) > bufferSize_) buffer_.pop_front();

        std::map<std::string, int> counts;
        for (const auto &l : buffer_) counts[l]++;

        std::string majorityLabel = buffer_.front();
        int majorityCount = 0;
        for (const auto &kv : counts) {
            if (kv.second > majorityCount) {
                majorityCount = kv.second;
                majorityLabel = kv.first;
            }
        }
        stabilityRatio = static_cast<float>(majorityCount) / static_cast<float>(buffer_.size());

        if (majorityLabel == stableLabel_) {
            stableCount_++;
        } else {
            stableLabel_ = majorityLabel;
            stableCount_ = 1;
            alreadyCommitted_ = false;
        }

        committed = false;
        if (stableCount_ >= stabilityFrames_ && stabilityRatio >= 0.6f &&
            majorityLabel != "UNKNOWN" && majorityLabel != "NOTHING" && majorityLabel != "IDLE" &&
            !alreadyCommitted_) {
            committed = true;
            alreadyCommitted_ = true;
        }

        return majorityLabel;
    }

    void Reset() {
        buffer_.clear();
        stableLabel_.clear();
        stableCount_ = 0;
        alreadyCommitted_ = false;
    }

private:
    std::deque<std::string> buffer_;
    int bufferSize_;
    float confidenceThreshold_;
    int stabilityFrames_;
    std::string stableLabel_;
    int stableCount_ = 0;
    bool alreadyCommitted_ = false;
};

class SentenceBuilder {
public:
    void AddCharacter(const std::string &label) {
        if (label == "SPACE") {
            sentence_ += " ";
            capitalizeNext_ = true;
        } else if (label == "DELETE") {
            if (!sentence_.empty()) sentence_.pop_back();
        } else if (label.size() == 1 && std::isalpha(static_cast<unsigned char>(label[0]))) {
            char c = capitalizeNext_ ? std::toupper(label[0]) : std::tolower(label[0]);
            sentence_ += c;
            capitalizeNext_ = false;
            letterFrequency_[label]++;
        }
        history_.push_back(label);
        if (history_.size() > 200) history_.pop_front();
    }

    void Clear() {
        sentence_.clear();
        capitalizeNext_ = true;
    }

    std::string Export() const {
        std::string path = kOutputDir + "/sentence_" + std::to_string(std::time(nullptr)) + ".txt";
        std::ofstream out(path);
        out << sentence_;
        return path;
    }

    const std::string &GetSentence() const { return sentence_; }
    const std::deque<std::string> &GetHistory() const { return history_; }

private:
    std::string sentence_;
    std::deque<std::string> history_;
    std::map<std::string, int> letterFrequency_;
    bool capitalizeNext_ = true;
};

class AnalyticsTracker {
public:
    AnalyticsTracker() : sessionStart_(std::time(nullptr)) {}

    void RecordFrame(float fps) {
        frameCount_++;
        fpsHistory_.push_back(fps);
        if (fpsHistory_.size() > 60) fpsHistory_.pop_front();
    }

    void RecordRecognition(const std::string &label) {
        recognitionEvents_++;
        if (label.size() == 1) letterFrequency_[label]++;
    }

    float AverageFps() const {
        if (fpsHistory_.empty()) return 0.0f;
        float sum = 0.0f;
        for (float v : fpsHistory_) sum += v;
        return sum / static_cast<float>(fpsHistory_.size());
    }

    long SessionDurationSeconds() const { return std::time(nullptr) - sessionStart_; }

    int FrameCount() const { return frameCount_; }
    int RecognitionEvents() const { return recognitionEvents_; }

private:
    std::time_t sessionStart_;
    std::deque<float> fpsHistory_;
    int frameCount_ = 0;
    int recognitionEvents_ = 0;
    std::map<std::string, int> letterFrequency_;
};

class UIRenderer {
public:
    UIRenderer(int width, int height) : width_(width), height_(height) {}

    void SetSize(int width, int height) {
        width_ = width;
        height_ = height;
    }

    void DrawGlassPanel(cv::Mat &frame, int x, int y, int w, int h, double alpha = 0.55) {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Rect(x, y, w, h), cv::Scalar(32, 24, 24), -1);
        cv::rectangle(overlay, cv::Rect(x, y, w, h), cv::Scalar(0, 140, 255), 1);
        cv::addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);
    }

    void DrawConfidenceBar(cv::Mat &frame, int x, int y, int w, int h, float confidence, const std::string &label) {
        cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(70, 60, 60), -1);
        int fillWidth = static_cast<int>(w * std::max(0.0f, std::min(1.0f, confidence)));
        cv::Scalar color = confidence > 0.7f ? cv::Scalar(140, 220, 80)
                          : confidence > 0.4f ? cv::Scalar(0, 140, 255)
                                               : cv::Scalar(90, 90, 240);
        cv::rectangle(frame, cv::Rect(x, y, fillWidth, h), color, -1);
        cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(210, 200, 200), 1);
        std::ostringstream oss;
        oss << label << " " << static_cast<int>(confidence * 100) << "%";
        cv::putText(frame, oss.str(), cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(245, 235, 235), 1, cv::LINE_AA);
    }

    void DrawLandmarks(cv::Mat &frame, const std::vector<cv::Point> &pts,
                        const std::vector<std::pair<int, int>> &connections) {
        for (const auto &c : connections) {
            cv::line(frame, pts[c.first], pts[c.second], cv::Scalar(255, 210, 0), 2, cv::LINE_AA);
        }
        cv::Scalar palette[5] = {
            cv::Scalar(80, 80, 255), cv::Scalar(60, 200, 255), cv::Scalar(90, 255, 120),
            cv::Scalar(255, 200, 90), cv::Scalar(255, 120, 200)
        };
        for (size_t i = 0; i < pts.size(); i++) {
            cv::circle(frame, pts[i], 5, palette[i % 5], -1, cv::LINE_AA);
            cv::circle(frame, pts[i], 6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    void DrawBoundingBox(cv::Mat &frame, int x1, int y1, int x2, int y2,
                          const std::string &label, float confidence) {
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(120, 255, 60), 2);
        std::ostringstream oss;
        oss << label << " " << static_cast<int>(confidence * 100) << "%";
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::rectangle(frame, cv::Point(x1, y1 - textSize.height - 12),
                      cv::Point(x1 + textSize.width + 10, y1), cv::Scalar(120, 255, 60), -1);
        cv::putText(frame, oss.str(), cv::Point(x1 + 5, y1 - 6), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(10, 20, 10), 2, cv::LINE_AA);
    }

    void DrawHud(cv::Mat &frame, float fps, float avgFps, float latencyMs, int frameCount,
                 int handCount, const std::string &trackingStatus, const std::string &detectionStatus,
                 const std::string &currentLabel, float confidence, float stability,
                 float handConfidence, const std::string &sentence, const std::string &historyPreview,
                 bool paused) {
        DrawGlassPanel(frame, 10, 10, 340, 190);
        cv::putText(frame, "SIGN LANGUAGE RECOGNITION", cv::Point(24, 38), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                    cv::Scalar(0, 140, 255), 2, cv::LINE_AA);

        std::ostringstream l1, l2, l3, l4, l5;
        l1 << "FPS: " << std::fixed << std::setprecision(1) << fps << "  AVG: " << avgFps;
        l2 << "Latency: " << latencyMs << " ms";
        l3 << "Frame: " << frameCount;
        l4 << "Hands: " << handCount << "  Tracking: " << trackingStatus;
        l5 << "Detection: " << detectionStatus;

        std::vector<std::string> lines = {l1.str(), l2.str(), l3.str(), l4.str(), l5.str()};
        for (size_t i = 0; i < lines.size(); i++) {
            cv::putText(frame, lines[i], cv::Point(24, 66 + static_cast<int>(i) * 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(245, 235, 235), 1, cv::LINE_AA);
        }

        int panelX = width_ - 380;
        DrawGlassPanel(frame, panelX, 10, 370, 210);
        cv::putText(frame, "RECOGNITION", cv::Point(panelX + 14, 38), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                    cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
        cv::putText(frame, "Current: " + currentLabel, cv::Point(panelX + 14, 68), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(140, 220, 80), 2, cv::LINE_AA);
        DrawConfidenceBar(frame, panelX + 14, 92, 340, 16, confidence, "Confidence");
        DrawConfidenceBar(frame, panelX + 14, 130, 340, 16, stability, "Stability");

        std::ostringstream hc;
        hc << "Hand confidence: " << static_cast<int>(handConfidence * 100) << "%";
        cv::putText(frame, hc.str(), cv::Point(panelX + 14, 194), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    cv::Scalar(245, 235, 235), 1, cv::LINE_AA);

        int bottomY = height_ - 110;
        DrawGlassPanel(frame, 10, bottomY, width_ - 20, 100);
        cv::putText(frame, "SENTENCE", cv::Point(24, bottomY + 26), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
        std::string displaySentence = sentence.size() > 90 ? sentence.substr(sentence.size() - 90) : sentence;
        cv::putText(frame, displaySentence, cv::Point(24, bottomY + 58), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(245, 235, 235), 2, cv::LINE_AA);
        cv::putText(frame, "History: " + historyPreview, cv::Point(24, bottomY + 84), cv::FONT_HERSHEY_SIMPLEX,
                    0.42, cv::Scalar(180, 170, 170), 1, cv::LINE_AA);

        cv::putText(frame,
                    "P Pause  S Screenshot  R Record  L Landmarks  K Skeleton  H HUD  C Clear  X Export  Q Exit",
                    cv::Point(24, height_ - 12), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(160, 150, 150), 1,
                    cv::LINE_AA);

        if (paused) {
            cv::putText(frame, "PAUSED", cv::Point(width_ / 2 - 80, height_ / 2), cv::FONT_HERSHEY_SIMPLEX,
                        1.4, cv::Scalar(90, 90, 240), 3, cv::LINE_AA);
        }
    }

private:
    int width_;
    int height_;
};

static const std::vector<std::pair<int, int>> kHandConnections = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {5, 9}, {9, 10}, {10, 11}, {11, 12},
    {9, 13}, {13, 14}, {14, 15}, {15, 16},
    {13, 17}, {17, 18}, {18, 19}, {19, 20},
    {0, 17}
};

class SignLanguageRecognitionApp {
public:
    SignLanguageRecognitionApp() : ui_(1280, 720), smoother_(12, 0.5f, 8) {
        fs::create_directories(kOutputDir);
        fs::create_directories(kScreenshotDir);
        fs::create_directories(kRecordingDir);
        fs::create_directories(kLogDir);

        auto config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(kGraphConfig);
        auto status = graph_.Initialize(config);
        if (!status.ok()) {
            std::cerr << "Failed to initialize MediaPipe graph: " << status.message() << std::endl;
            std::exit(1);
        }

        auto landmarksPollerOrStatus = graph_.AddOutputStreamPoller("landmarks");
        auto handednessPollerOrStatus = graph_.AddOutputStreamPoller("handedness");
        if (!landmarksPollerOrStatus.ok() || !handednessPollerOrStatus.ok()) {
            std::cerr << "Failed to add output stream pollers" << std::endl;
            std::exit(1);
        }
        landmarksPoller_ = std::move(landmarksPollerOrStatus.value());
        handednessPoller_ = std::move(handednessPollerOrStatus.value());

        auto startStatus = graph_.StartRun({});
        if (!startStatus.ok()) {
            std::cerr << "Failed to start MediaPipe graph: " << startStatus.message() << std::endl;
            std::exit(1);
        }

        capture_.open(0);
        if (!capture_.isOpened()) {
            std::cerr << "Webcam not detected. Please connect a webcam and rerun." << std::endl;
            std::exit(1);
        }
        capture_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        lastFrameTime_ = std::chrono::steady_clock::now();
    }

    void Run() {
        cv::namedWindow("Real-Time Sign Language Recognition", cv::WINDOW_NORMAL);
        cv::resizeWindow("Real-Time Sign Language Recognition", 1280, 720);

        cv::Mat frame;
        while (true) {
            if (!paused_) {
                capture_ >> frame;
                if (frame.empty()) continue;
                cv::Mat display = ProcessFrame(frame);
                cv::imshow("Real-Time Sign Language Recognition", display);
            } else {
                capture_ >> frame;
                if (!frame.empty()) {
                    cv::flip(frame, frame, 1);
                    ui_.DrawHud(frame, 0, analytics_.AverageFps(), 0, analytics_.FrameCount(), 0, "Paused",
                                "Paused", "PAUSED", 0, 0, 0, sentenceBuilder_.GetSentence(), "", true);
                    cv::imshow("Real-Time Sign Language Recognition", frame);
                }
            }

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;
            if (key == 'p') paused_ = !paused_;
            if (key == 's') SaveScreenshot(frame);
            if (key == 'l') showLandmarks_ = !showLandmarks_;
            if (key == 'h') showHud_ = !showHud_;
            if (key == 'c') sentenceBuilder_.Clear();
            if (key == 'x') std::cout << "Sentence exported to: " << sentenceBuilder_.Export() << std::endl;
        }

        Shutdown();
    }

private:
    cv::Mat ProcessFrame(cv::Mat frame) {
        cv::flip(frame, frame, 1);
        int width = frame.cols;
        int height = frame.rows;
        ui_.SetSize(width, height);

        mediapipe::ImageFrame inputImage(mediapipe::ImageFormat::SRGB, width, height,
                                          mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat inputMat = mediapipe::formats::MatView(&inputImage);
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        rgb.copyTo(inputMat);

        size_t frameTimestampUs = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        auto packet = mediapipe::Adopt(new mediapipe::ImageFrame(std::move(inputImage)))
                          .At(mediapipe::Timestamp(frameTimestampUs));
        auto addStatus = graph_.AddPacketToInputStream("input_video", packet);
        if (!addStatus.ok()) {
            return frame;
        }

        std::string currentLabel = "IDLE";
        float confidence = 0.0f;
        float stability = 0.0f;
        float handConfidence = 0.0f;
        int handCount = 0;
        std::string detectionStatus = "No Hand";
        std::string trackingStatus = "Idle";
        bool committed = false;

        mediapipe::Packet landmarksPacket;
        mediapipe::Packet handednessPacket;
        bool gotLandmarks = landmarksPoller_.QueueSize() > 0 && landmarksPoller_.Next(&landmarksPacket);
        bool gotHandedness = handednessPoller_.QueueSize() > 0 && handednessPoller_.Next(&handednessPacket);

        if (gotLandmarks) {
            const auto &landmarkLists = landmarksPacket.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            if (!landmarkLists.empty()) {
                handCount = static_cast<int>(landmarkLists.size());
                detectionStatus = "Tracking";
                trackingStatus = "Active";
                noHandSince_ = -1.0;
                autoSpaceInserted_ = false;

                const auto &firstHand = landmarkLists[0];
                std::vector<Landmark3D> lm;
                for (int i = 0; i < firstHand.landmark_size(); i++) {
                    const auto &p = firstHand.landmark(i);
                    lm.push_back({p.x(), p.y(), p.z()});
                }

                if (gotHandedness) {
                    const auto &handednessList = handednessPacket.Get<std::vector<mediapipe::ClassificationList>>();
                    if (!handednessList.empty() && handednessList[0].classification_size() > 0) {
                        handConfidence = handednessList[0].classification(0).score();
                    }
                }

                Prediction pred = ClassifyGesture(lm);
                currentLabel = smoother_.Update(pred.label, pred.confidence, stability, committed);
                confidence = pred.confidence;

                std::vector<cv::Point> pts;
                float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
                for (const auto &p : lm) {
                    int px = static_cast<int>(p.x * width);
                    int py = static_cast<int>(p.y * height);
                    pts.push_back(cv::Point(px, py));
                    minX = std::min(minX, static_cast<float>(px));
                    minY = std::min(minY, static_cast<float>(py));
                    maxX = std::max(maxX, static_cast<float>(px));
                    maxY = std::max(maxY, static_cast<float>(py));
                }

                if (showLandmarks_) ui_.DrawLandmarks(frame, pts, kHandConnections);
                ui_.DrawBoundingBox(frame, static_cast<int>(minX) - 20, static_cast<int>(minY) - 20,
                                     static_cast<int>(maxX) + 20, static_cast<int>(maxY) + 20,
                                     currentLabel, confidence);
            }
        }

        if (handCount == 0) {
            smoother_.Reset();
            double nowSec = static_cast<double>(std::time(nullptr));
            if (noHandSince_ < 0) {
                noHandSince_ = nowSec;
            } else if (!autoSpaceInserted_ && nowSec - noHandSince_ > 0.7) {
                const std::string &sentence = sentenceBuilder_.GetSentence();
                if (!sentence.empty() && sentence.back() != ' ') {
                    sentenceBuilder_.AddCharacter("SPACE");
                }
                autoSpaceInserted_ = true;
            }
        }

        if (committed) {
            sentenceBuilder_.AddCharacter(currentLabel);
            analytics_.RecordRecognition(currentLabel);
        }

        auto now = std::chrono::steady_clock::now();
        float frameTimeMs = std::chrono::duration<float, std::milli>(now - lastFrameTime_).count();
        float fps = frameTimeMs > 0 ? 1000.0f / frameTimeMs : 0.0f;
        lastFrameTime_ = now;
        analytics_.RecordFrame(fps);
        frameCount_++;

        std::string historyPreview;
        int count = 0;
        for (auto it = sentenceBuilder_.GetHistory().rbegin();
             it != sentenceBuilder_.GetHistory().rend() && count < 15; ++it, ++count) {
            historyPreview = (it->size() == 1 ? *it : "_") + historyPreview;
        }

        if (showHud_) {
            ui_.DrawHud(frame, fps, analytics_.AverageFps(), frameTimeMs, frameCount_, handCount,
                        trackingStatus, detectionStatus, currentLabel, confidence, stability,
                        handConfidence, sentenceBuilder_.GetSentence(), historyPreview, false);
        }

        return frame;
    }

    void SaveScreenshot(const cv::Mat &frame) {
        if (frame.empty()) return;
        std::string path = kScreenshotDir + "/screenshot_" + std::to_string(std::time(nullptr)) + ".png";
        cv::imwrite(path, frame);
        std::cout << "Screenshot saved to " << path << std::endl;
    }

    void Shutdown() {
        capture_.release();
        cv::destroyAllWindows();
        graph_.CloseInputStream("input_video").IgnoreError();
        graph_.WaitUntilDone().IgnoreError();

        std::string summaryPath = kLogDir + "/session_summary_" + std::to_string(std::time(nullptr)) + ".json";
        std::ofstream out(summaryPath);
        out << "{\n";
        out << "  \"session_duration_seconds\": " << analytics_.SessionDurationSeconds() << ",\n";
        out << "  \"average_fps\": " << analytics_.AverageFps() << ",\n";
        out << "  \"total_frames\": " << analytics_.FrameCount() << ",\n";
        out << "  \"recognition_events\": " << analytics_.RecognitionEvents() << ",\n";
        out << "  \"final_sentence\": \"" << sentenceBuilder_.GetSentence() << "\"\n";
        out << "}\n";
        std::cout << "Session summary saved to " << summaryPath << std::endl;
    }

    mediapipe::CalculatorGraph graph_;
    mediapipe::OutputStreamPoller landmarksPoller_ = mediapipe::OutputStreamPoller();
    mediapipe::OutputStreamPoller handednessPoller_ = mediapipe::OutputStreamPoller();
    cv::VideoCapture capture_;
    UIRenderer ui_;
    PredictionSmoother smoother_;
    SentenceBuilder sentenceBuilder_;
    AnalyticsTracker analytics_;

    bool paused_ = false;
    bool showLandmarks_ = true;
    bool showHud_ = true;
    double noHandSince_ = -1.0;
    bool autoSpaceInserted_ = true;
    int frameCount_ = 0;
    std::chrono::steady_clock::time_point lastFrameTime_;
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    SignLanguageRecognitionApp app;
    app.Run();
    return 0;
}