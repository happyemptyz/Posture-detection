import logging,  cv2, time, numpy as np, mediapipe as mp, tkinter as tk, sys, mysql.connector, os, subprocess, webbrowser, uuid
from datetime import datetime 
from tkinter import simpledialog, messagebox

class Application(tk.Tk):
    def __init__(self, pose_detection): 
        super().__init__()
        self.pose_detection = pose_detection
        self.title('Squat Posture Analysis System')
        self.geometry('900x600')
        self.create_widgets()
    def create_widgets(self):
        button_width = 20
        button_height = 2
        tk.Button(self, text="直接進行深蹲判斷", command=self.start_pose_detection, width=button_width, height=button_height).pack(pady=30) # 創建四個按鈕
        tk.Button(self, text="開始進行負重深蹲", command=self.start_weighted_squat, width=button_width, height=button_height).pack(pady=30)
        tk.Button(self, text="查看數據分析",    command=self.view_data_analysis,   width=button_width, height=button_height).pack(pady=30)
        tk.Button(self, text="離開",           command=self.quit_application,     width=button_width, height=button_height).pack(pady=30)
    def start_pose_detection(self):
        self.hide()
        self.pose_detection.start_new_session()
        try: 
            main(self.pose_detection)
        except Exception as e:
            messagebox.showerror("錯誤", f"深蹲姿勢判斷發生錯誤: {e}")
        self.show()
    def start_weighted_squat(self):
        self.hide() 
        gender = self.ask_for_gender() 
        if gender: 
            self.pose_detection.gender = gender
        else:
            messagebox.showerror("錯誤", "沒有選擇性別！")
            self.show() 
            return
        weight = simpledialog.askstring("輸入", "請輸入要負重的重量:")
        if weight is not None:
            try:
                weight_float = float(weight)
                self.pose_detection.weight = weight_float
                print(f"self.pose_detection.weight: {self.pose_detection.weight}")
            except ValueError:
                messagebox.showerror("錯誤", "請輸入有效的數字！")
                return
        else:
            messagebox.showerror("錯誤", "沒有輸入負重！")
            return
        body_weight_str = simpledialog.askstring("輸入", "請輸入你的體重:")
        if body_weight_str is not None:
            try:
                body_weight_float = float(body_weight_str)
                self.pose_detection.body_weight = body_weight_float
            except ValueError:
                messagebox.showerror("錯誤", "請輸入有效的數字！")
                return
        else:
            messagebox.showerror("錯誤", "沒有輸入體重！")
            return 
        response = messagebox.askyesno("選項", "1. 進入深蹲判判斷程式\n2. 返回主畫面") 
        if response:
            self.start_pose_detection()
        else: 
            self.show()        
    def ask_for_gender(self):
        dialog = tk.Toplevel(self)
        dialog.title("選擇性別")
        dialog.geometry('900x600') 
        tk.Label(dialog, text="請選擇你的性別：", font=("Arial", 20)).pack(pady=10)
        selected_gender = tk.StringVar()
        button_width = 20
        button_height = 2
        font_size = ("Arial", 20)
        tk.Button(dialog, text="Male", command=lambda: selected_gender.set('male'),
                  height=button_height, width=button_width, font=font_size).pack(fill='x', expand=True, pady=10)
        tk.Button(dialog, text="Female", command=lambda: selected_gender.set('female'),
                  height=button_height, width=button_width, font=font_size).pack(fill='x', expand=True, pady=10)
        dialog.wait_variable(selected_gender)
        dialog.destroy()
        return selected_gender.get()
    def view_data_analysis(self):
        url = 'http://localhost:3000/d/bdjni9xiebdoge/sessions-table?orgId=1'
        try:
            chrome_path = "open -a '/Applications/Google Chrome 2.app' %s"
            webbrowser.get(chrome_path).open(url)
        except Exception as e:
            messagebox.showerror("錯誤", f"無法打開網頁: {e}")
    def quit_application(self):
        self.quit()
    def hide(self): 
        self.withdraw() 
    def show(self):
        self.update() 
        self.deiconify()
def gui_main():
    detector = PoseDetection() 
    app = Application(detector)
    app.mainloop() 





class PoseDetection(): 
    def __init__(self, image_mode=False, model_compl=1, smooth_lm=True, enable_seg=False, smooth_seg=True, min_detection_conf=0.5, min_tracking_conf=0.5, flip_frame=False, weight=0):
        self.setup_logging()
        self.weight = weight
        self.body_weight = 0
        self.one_rm_kg = 0 
        self.session_id = str(uuid.uuid4())
        self.image_mode = image_mode
        self.model_compl = model_compl
        self.smooth_lm = smooth_lm
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.flip_frame = flip_frame
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.initialize_landmarks_info()  
        self.radius = 20
        self.COLORS = {'blue': (0, 127, 255), 'red': (255, 50, 50), 'green': (0, 255, 127), 'light_green': (100, 233, 127), 'yellow': (255, 255, 0), 'magenta': (255, 0, 255), 'white': (255, 255, 255), 'cyan': (0, 255, 255), 'light_blue': (102, 204, 255)}
        self.dict_features = {}
        self.left_features = {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27, 'foot': 31}
        self.right_features = {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28, 'foot': 32}
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0
        self._ANGLE_HIP_KNEE_VERT = {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (70, 95)}
        self.thresholds = {'HIP_KNEE_VERT': self._ANGLE_HIP_KNEE_VERT,'HIP_THRESH': [10, 50],'ANKLE_THRESH': 45,'KNEE_THRESH': [50, 70, 95],'OFFSET_THRESH': 35.0,'INACTIVE_THRESH': 15.0,'CNT_FRAME_THRESH': 50}
        self.state_tracker = {'state_seq': [],
                      'start_inactive_time': time.perf_counter(),
                      'start_inactive_time_front': time.perf_counter(),
                      'INACTIVE_TIME': 0.0,
                      'INACTIVE_TIME_FRONT': 0.0,
                      'DISPLAY_TEXT': np.full((4,), False),
                      'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),
                      'LOWER_HIPS': False,
                      'Incorrect_POSTURE': False,
                      'prev_state': None,
                      'curr_state': None,
                      'SQUAT_COUNT': 0,
                      'IMPROPER_SQUAT': 0}
        self.FEEDBACK_ID_MAP = {0: ('BEND BACKWARDS', 125, (0, 0, 255)), 1: ('BEND FORWARD', 125, (0, 0, 255)), 2: ('KNEE FALLING OVER TOE', 170, (0, 0, 255)), 3: ('SQUAT TOO DEEP', 215, (0, 0, 255))}
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.image_mode, self.model_compl, self.smooth_lm, self.enable_seg, self.smooth_seg, self.min_detection_conf, self.min_tracking_conf)
        self.pose = mp.solutions.pose.Pose(static_image_mode=image_mode, model_complexity=model_compl, smooth_landmarks=smooth_lm, enable_segmentation=enable_seg, smooth_segmentation=smooth_seg, min_detection_confidence=min_detection_conf, min_tracking_confidence=min_tracking_conf)
        self.landmark_names = { 0: 'nose',           1: 'left eye inner',  2: 'left eye',        3: 'left eye outer', 4: 'right eye inner', 5: 'right eye',    6: 'right eye outer',7: 'left ear',        8: 'right ear',
                                9: 'mouth left',     10: 'mouth right',    11: 'left shoulder', 12: 'right shoulder',13: 'left elbow',     14: 'right elbow', 15: 'left wrist',    16: 'right wrist',    17: 'left pinkie',
                                18: 'right pinkie',  19: 'left index',     20: 'right index',   21: 'left thumb',    22: 'right thumb',    23: 'left hip',    24: 'right hip',     25: 'left knee',      26: 'right knee', 
                                27: 'left ankle',    28: 'right ankle',    29: 'left heel',     30: 'right heel',    31: 'left foot index',32: 'right foot index',}
        self.landmarks_info = {}  
        self.initialize_landmarks_info() 
    def start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.end_time = None 
        logging.info(f"New session started with ID: {self.session_id}")        
    def end_session(self):
        self.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        start_dt = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S.%f')
        end_dt = datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S.%f')
        self.total_time = (end_dt - start_dt).total_seconds() 
        self.calculate_correct_incorrect_times()
        self.get_gender_and_accuracy()
        self.get_additional_info()
        self.log_session_to_database() 
    def get_additional_info(self):
        connection = self.connect_to_mysql()
        if connection is not None:
            try:
                cursor = connection.cursor()
                sql = """
                    SELECT one_rm_kg, level_1_to_7 
                    FROM squats 
                    WHERE session_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 1"""
                val = (self.session_id,)
                cursor.execute(sql, val)
                result = cursor.fetchone()
                if result:
                    self.one_rm_kg, self.level_1_to_7 = result
                else:
                    self.one_rm_kg, self.level_1_to_7 = 0, 'unknown'
            except mysql.connector.Error as error:
                logging.error(f"Failed to retrieve additional info: {error}")
            finally:
                cursor.close()
                connection.close()
    def calculate_correct_incorrect_times(self):
        connection = self.connect_to_mysql()
        if connection is not None:
            try:
                cursor = connection.cursor()
                sql = "SELECT MAX(correct_count), MAX(incorrect_count) FROM squats WHERE session_id = %s"
                val = (self.session_id,)
                cursor.execute(sql, val)
                result = cursor.fetchone()
                self.correct_times, self.incorrect_times = result
            except mysql.connector.Error as error:
                logging.error(f"Failed to calculate correct/incorrect times: {error}")
            finally:
                cursor.close()
                connection.close()
    def get_gender_and_accuracy(self):
        connection = self.connect_to_mysql()
        if connection is not None:
            try:
                cursor = connection.cursor()
                sql = "SELECT gender, accuracy FROM squats WHERE session_id = %s ORDER BY timestamp DESC LIMIT 1"
                val = (self.session_id,)
                cursor.execute(sql, val)
                result = cursor.fetchone()
                self.gender, self.accuracy_percentage = result
            except mysql.connector.Error as error:
                logging.error(f"Failed to retrieve gender and accuracy: {error}")
            finally:
                cursor.close()
                connection.close()
    def log_session_to_database(self):
        connection = self.connect_to_mysql()
        if connection is not None:
            try:
                cursor = connection.cursor()
                sql = """
                INSERT INTO sessions (
                    start_time, end_time, session_id, total_time, correct_times, incorrect_times, 
                    gender, accuracy_percentage, 1RM_kg, level_1_to7
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """           
                val = (self.start_time, self.end_time, self.session_id, self.total_time, self.correct_times, self.incorrect_times, self.gender, self.accuracy_percentage, self.one_rm_kg, self.level_1_to_7)
                cursor.execute(sql, val)
                connection.commit()
                logging.info("Session logged successfully with total time, correct times, and incorrect times.")
            except mysql.connector.Error as error:
                logging.error(f"Failed to log session: {error}")
            finally:
                cursor.close()
                connection.close()

    def update_one_rm(self, total_attempts): 
        if total_attempts > 0:
            self.one_rm_kg = self.weight * (1 + (total_attempts * 0.0333))
        else:
            self.one_rm_kg = 0
    def determine_level(self, one_rm_kg, body_weight, gender): 
        levels = {'male': {  55: [74, 98, 110, 122, 160, 169, 200],
                             61: [81, 109, 126, 143, 169, 194, 220],
                             67: [86, 113, 136, 158, 189, 201, 232],
                             73: [90, 118, 144, 168, 202, 223, 242],
                             81: [95, 127, 155, 183, 218, 237, 257],
                             89: [102, 138, 167, 196, 231, 252, 282],
                             96: [103, 141, 172, 202, 238, 267, 289],
                             102: [110, 146, 176, 207, 245, 273, 298],
                             109: [113, 149, 180, 210, 252, 279, 303],
                             500: [117, 155, 186, 217, 263, 290, 316]},
                 'female': { 45: [46, 60, 71, 83, 94, 111, 126],
                             49: [49, 65, 76, 90, 102, 120, 137],
                             55: [57, 76, 88, 98, 116, 138, 158],
                             59: [60, 80, 95, 109, 131, 149, 166],
                             64: [63, 84, 98, 112, 134, 155, 175],
                             71: [68, 92, 106, 119, 141, 165, 189],
                             76: [71, 95, 109, 123, 147, 173, 201],
                             81: [76, 102, 114, 126, 154, 185, 210],
                             87: [81, 109, 119, 130, 156, 189, 217],
                             500: [74, 112, 123, 133, 168, 195, 223]}}
        weight_classes = sorted(levels[gender].keys())
        weight_class = max(wc for wc in weight_classes if wc <= body_weight)
        for level, required_weight in enumerate(levels[gender][weight_class], start=1):
            if one_rm_kg <= required_weight:
                return f'level_{level - 1 if level > 1 else level}'
            elif one_rm_kg == required_weight:
                return f'level_{level}'
        return 'level_7'   
    def generate_error_details(self):
        feedback_messages = []
        if self.state_tracker['DISPLAY_TEXT'][0]: 
            feedback_messages.append('BEND BACKWARDS')
        if self.state_tracker['DISPLAY_TEXT'][1]: 
            feedback_messages.append('BEND FORWARD')
        if self.state_tracker['DISPLAY_TEXT'][2]: 
            feedback_messages.append('KNEE FALLING OVER TOE')
        if self.state_tracker['DISPLAY_TEXT'][3]:  
            feedback_messages.append('SQUAT TOO DEEP')
        if self.state_tracker['LOWER_HIPS']: 
            feedback_messages.insert(0, 'LOWER YOUR HIPS') 
        return ', '.join(feedback_messages)
    
    def initialize_landmarks_info(self):
        all_landmark_keys = ['nose',           'left_eye_inner',   'left_eye',        'left_eye_outer',  'right_eye_inner', 'right_eye', 
                             'right_eye_outer', 'left_ear',         'right_ear',       'mouth_left',      'mouth_right',     'left_shoulder', 
                             'right_shoulder',  'left_elbow',       'right_elbow',     'left_wrist',      'right_wrist',     'left_pinky', 
                             'right_pinky',    'left_index',       'right_index',     'left_thumb',      'right_thumb',     'left_hip', 
                             'right_hip',      'left_knee',        'right_knee',      'left_ankle',      'right_ankle',     'left_heel', 
                             'right_heel',     'left_foot_index',  'right_foot_index']
        self.landmarks_info = {key: (None, None) for key in all_landmark_keys}
    @staticmethod
    def setup_logging():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('output_log.txt')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def connect_to_mysql(self):
        try:
            return mysql.connector.connect(host="localhost", user="root", password="happyemptyz", database="fitness_db", auth_plugin='mysql_native_password')
        except mysql.connector.Error as error:
            logging.error(f"Failed to connect to database: {error}")
            return None

    def log_landmark(self, id, timestamp, landmark_index, x, y):
        landmark_name = self.landmark_names.get(landmark_index, f"Unknown landmark {landmark_index}")
        self.landmarks_info[landmark_name] = (x, y)

    def flush_landmarks_to_database(self):
        logging.info(f"準備寫入資料庫前的 self.weight: {self.weight}")
        logging.info("Flushing landmarks to database...")
        connection = self.connect_to_mysql()
        
        if connection is not None:
            try:
                cursor = connection.cursor()
                formatted_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                gender = self.gender if hasattr(self, 'gender') and self.gender in ['male', 'female'] else 'unknown'
                error_details = self.generate_error_details() 
                total_attempts = self.state_tracker['SQUAT_COUNT'] + self.state_tracker['IMPROPER_SQUAT']
                self.update_one_rm(total_attempts)
                level = self.determine_level(self.one_rm_kg, self.body_weight, gender)
                accuracy = (self.state_tracker['SQUAT_COUNT'] / total_attempts) * 100 if total_attempts > 0 else 0
                detected_landmarks = {k: v for k, v in self.landmarks_info.items() if v[0] is not None}
                if detected_landmarks: 
                  sql_columns = ", ".join([f"`{name.replace(' ', '_')}_x_coord`, `{name.replace(' ', '_')}_y_coord`" for name in detected_landmarks.keys()])
                  sql_values_placeholders = ", ".join(["%s"] * (len(detected_landmarks) * 2))
                  sql = f"INSERT INTO squats (session_id, timestamp, gender, correct_count, incorrect_count, total_attempts, accuracy, error_details, one_rm_kg, level_1_to_7, {sql_columns}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {sql_values_placeholders})"
                  val = [self.session_id, formatted_timestamp, gender, self.state_tracker['SQUAT_COUNT'], self.state_tracker['IMPROPER_SQUAT'], total_attempts, accuracy, error_details, self.one_rm_kg, level] + [coord for coords in detected_landmarks.values() for coord in coords]
                  cursor.execute(sql, val)
                  connection.commit()
                  logging.info("Data inserted successfully")
                else:
                    logging.info("No landmarks detected, skipping database insert.")
            except mysql.connector.Error as error:
                logging
            finally:
                cursor.close()
                connection.close()
            self.initialize_landmarks_info()  # 重置 landmarks_info
        else:
            logging.error("Failed to connect to the database.")


    def draw_rounded_rect(self, frame, rect_start, rect_end, corner_width, box_color):
        x1, y1 = rect_start
        x2, y2 = rect_end
        w = corner_width
        cv2.rectangle(frame, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)        # draw filled rectangles
        cv2.rectangle(frame, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
        cv2.rectangle(frame, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
        cv2.rectangle(frame, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
        cv2.rectangle(frame, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)
        cv2.ellipse(frame, (x1 + w, y1 + w), (w, w), angle=0, startAngle=-90, endAngle=-180, color=box_color, thickness=-1)        # draw filled ellipses
        cv2.ellipse(frame, (x2 - w, y1 + w), (w, w), angle=0, startAngle=0, endAngle=-90, color=box_color, thickness=-1)
        cv2.ellipse(frame, (x1 + w, y2 - w), (w, w), angle=0, startAngle=90, endAngle=180, color=box_color, thickness=-1)
        cv2.ellipse(frame, (x2 - w, y2 - w), (w, w), angle=0, startAngle=0, endAngle=90, color=box_color, thickness=-1)
        return frame

    def draw_dotted_line(self, frame, lm_coord, start, end, line_color):
        pix_step = 0
        for i in range(start, end+1, 8):
            cv2.circle(frame, (lm_coord[0], i+pix_step),
                       2, line_color, -1, lineType=cv2.LINE_AA)
        return frame

    def draw_text(self,
        frame,
        msg,
        width=8,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(255, 255, 0),
        text_color_bg=(0, 0, 0),
        box_offset=(20, 10)):

        offset = box_offset
        x, y = pos
        text_size, _ = cv2.getTextSize(
            msg, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(p - o for p, o in zip(pos, offset))
        rec_end = tuple(m + n - o for m, n,
                        o in zip((x + text_w, y + text_h), offset, (25, 0)))

        frame = self.draw_rounded_rect(
            frame, rec_start, rec_end, width, text_color_bg)

        cv2.putText(
            frame,
            msg,
            (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,  )
        return text_size

    def find_angle(self, p1, p2, ref_pt=np.array([0, 0])):
        p1_ref = p1 - ref_pt
        p2_ref = p2 - ref_pt

        cos_theta = (np.dot(p1_ref, p2_ref)) / \
            (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        degree = int(180 / np.pi) * theta
        #print(f"Angle between points: {int(degree)} degrees")
        logging.info(f"Angle between points: {int(degree)} degrees") #f"點之間的角度：{int(degree)}度"
        return int(degree)

    def get_landmark_array(self, pose_landmark, key, frame_width, frame_height):

        denorm_x = int(pose_landmark[key].x * frame_width)
        denorm_y = int(pose_landmark[key].y * frame_height)
        logging.info(f"Landmark {key}: ({denorm_x}, {denorm_y})") 
        #print(f"Landmark {key}: ({denorm_x}, {denorm_y})") 
        return np.array([denorm_x, denorm_y])

    def get_landmark_features(self, kp_results, dict_features, feature, frame_width, frame_height):

        if feature == 'nose':
            return self.get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

        elif feature == 'left' or 'right':
            shldr_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
            elbow_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
            wrist_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
            hip_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['hip'], frame_width, frame_height)
            knee_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['knee'], frame_width, frame_height)
            ankle_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
            foot_coord = self.get_landmark_array(
                kp_results, dict_features[feature]['foot'], frame_width, frame_height)

            return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord

        else:
            raise ValueError(
                "feature needs to be either 'nose', 'left' or 'right")

    def _get_state(self, knee_angle):

        knee = None

        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3
        return f's{knee}' if knee else None

    def _update_state_sequence(self, state):

        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')) == 0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 1)):
                self.state_tracker['state_seq'].append(state)

        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):

        if lower_hips_disp:
            self.draw_text(
                frame,
                'LOWER YOUR HIPS',
                pos=(30, 80),
                text_color=(255, 255, 255),
                font_scale=0.6,
                text_color_bg=(0, 0, 255)
            )

        for idx in np.where(c_frame)[0]:
            self.draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=dict_maps[idx][2])
        return frame

    def process(self, frame):
        # 獲取圖像尺寸
        frame_height, frame_width, _ = frame.shape
        # Process the image with MediaPipe Pose
        results = self.pose.process(frame)
        # Check if any landmarks are detected
        if results.pose_landmarks:
            for key, landmark in enumerate(results.pose_landmarks.landmark):
                denorm_x = int(landmark.x * frame_width)
                denorm_y = int(landmark.y * frame_height)
                current_timestamp = time.time()
                self.log_landmark(id=key, timestamp=current_timestamp, landmark_index=key, x=denorm_x, y=denorm_y)
            self.flush_landmarks_to_database()
        play_sound = None
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        # Process the image.
        self.keypoints = self.pose.process(frame)
        if self.keypoints.pose_landmarks:
            self.ps_lm = self.keypoints.pose_landmarks
            nose_coord = self.get_landmark_features(
                self.ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                self.get_landmark_features(
                    self.ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                self.get_landmark_features(
                    self.ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)
            offset_angle = self.find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)
            self.flush_landmarks_to_database()
            if offset_angle > self.thresholds['OFFSET_THRESH']:
                display_inactivity = False
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time
                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True
                cv2.circle(frame, nose_coord,        7, self.COLORS['white'],   -1)
                cv2.circle(frame, left_shldr_coord,  7, self.COLORS['yellow'],  -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 90),
                    #             self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                self.draw_text(
                    frame,
                    "Correct: " + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 255),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0) )
                self.draw_text(
                    frame,
                    "Incorrect: " + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 255),
                    font_scale=0.7,
                    text_color_bg=(0, 0, 255), )
                self.draw_text(
                    frame,
                    'Camera not aligned properly!!!',#相機未正確對準！
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 255),
                    font_scale=0.7,
                    text_color_bg=(190, 190, 190), )
                self.draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),#偏移角度
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 255),
                    font_scale=0.7,
                    text_color_bg=(190, 190, 190),)
                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None
            # Camera is aligned properly.
            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                dist_l_sh_hip = abs(left_foot_coord[1] - left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]
                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None
                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord
                    multiplier = -1
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord
                    multiplier = 1

                # ------------------- Verical Angle calculation --------------
                hip_vertical_angle = self.find_angle(
                    shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90+multiplier*hip_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                self.draw_dotted_line(
                    frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])

                knee_vertical_angle = self.find_angle(
                    hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20),
                            angle=0, startAngle=-90, endAngle=-90-multiplier*knee_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                self.draw_dotted_line(
                    frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])
                ankle_vertical_angle = self.find_angle(
                    knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                logging.info(f"Knee angle: {knee_vertical_angle}, Ankle angle: {ankle_vertical_angle}") 
                #print(f"Knee angle: {knee_vertical_angle}, Ankle angle: {ankle_vertical_angle}") 
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*ankle_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                self.draw_dotted_line(
                    frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])
                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['Incorrect_POSTURE']:
                        self.state_tracker['SQUAT_COUNT'] += 1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'Incorrect'

                    elif self.state_tracker['Incorrect_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'Incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['Incorrect_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True

                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['DISPLAY_TEXT'][1] = True

                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                       self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['LOWER_HIPS'] = True

                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['Incorrect_POSTURE'] = True

                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['Incorrect_POSTURE'] = True

                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - \
                        self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------

                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1

                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting COUNTERS due to inactivity!!!', (10, frame_height - 20), self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x,
                            hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x,
                            knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x,
                            ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                self.draw_text(
                    frame,
                    "Correct: " + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 255),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                self.draw_text(
                    frame,
                    "Incorrect: " + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 255),
                    font_scale=0.7,
                    text_color_bg=(0, 0, 255),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.thresholds['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - \
                self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 25), self.font, 0.7, self.COLORS['blue'], 2)
                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            self.draw_text(
                frame,
                "Correct: " + str(self.state_tracker['SQUAT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 255),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            self.draw_text(
                frame,
                "Incorrect: " + str(self.state_tracker['IMPROPER_SQUAT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(0, 0, 255),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['Incorrect_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound
     

def main(detector):  
    # 使用第一個攝像頭
    cap = cv2.VideoCapture(0)
    # 檢查攝像頭是否正確開啟
    if not cap.isOpened():
        logging.error("Failed to open video capture device.")
        return
    # 定義影片編碼器和創建 VideoWriter 對象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定mp4格式的編碼
    out = cv2.VideoWriter('test1.mp4', fourcc, 10.0, (640, 480))  # 指定文件名，編碼器，幀率和解析度
    pTime = 0  # 初始化之前的時間，用於FPS計算
    while True:
        success, frame = cap.read()  # 從攝像頭讀取一幀
        if not success:
            logging.error("Failed to read frame from video capture.")
            break
        frame = cv2.resize(frame, (640, 480))  # 調整幀的大小
        try:
            img = detector.process(frame)  # 對讀取的幀進行姿態檢測處理
        except Exception as e:
            logging.exception("Error in process method")
            break
        cTime = time.time()  # 獲取當前時間
        fps = 1 / (cTime - pTime)  # 計算FPS
        pTime = cTime  # 更新之前時間
        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)  # 在畫面上顯示FPS
        cv2.imshow('CAM', frame)  # 顯示處理後的視頻幀
        out.write(frame)  # 將幀寫入輸出文件
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 則退出
            detector.end_session()  # 調用 end_session 方法
            break
    # 釋放資源和關閉窗口
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gui_main()
