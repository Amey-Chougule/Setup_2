import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import time
from twilio.rest import Client
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Twilio credentials (use environment variables in production)
TWILIO_ACCOUNT_SID = 'ACb8716c6019b4ffbdd89ca2deb160e11f'
TWILIO_AUTH_TOKEN = '9467b6e665c1d5c388beeb2c4f7c6e1e'
TWILIO_PHONE_NUMBER = '+18329650334'  # Your Twilio number

DATASET_DIR = 'dataset'
CSV_FILE = 'attendance.csv'
STUDENT_LIST_FILE = 'students_list.csv'
os.makedirs(DATASET_DIR, exist_ok=True)

def send_sms(to, message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to
        )
        print(f"[SMS SENT] -> {to}")
    except Exception as e:
        print(f"[ERROR] Failed to send SMS to {to}: {e}")

def load_known_faces():
    encodings = []
    metadata = []
    for file in os.listdir(DATASET_DIR):
        if file.endswith('.npy'):
            data = np.load(os.path.join(DATASET_DIR, file), allow_pickle=True).item()
            encodings.append(data['encoding'])
            metadata.append(data['info'])
    return encodings, metadata

def save_face_encoding(encoding, info):
    filename = f"{info['name']}_{info['roll']}.npy"
    np.save(os.path.join(DATASET_DIR, filename), {'encoding': encoding, 'info': info})
    print("[INFO] Face data saved!")

def is_attendance_marked(name):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        date = datetime.now().strftime("%Y-%m-%d")
        existing_record = df[(df['Name'] == name) & (df['Date'] == date)]
        return not existing_record.empty
    return False

def register_face(name_entry, roll_entry, mobile_entry, gender_var, class_entry, division_entry, register_frame):
    cap = cv2.VideoCapture(1)
    print("[INFO] Position your face and press 's' to capture.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Register - Press 's' to capture", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            if boxes:
                encoding = face_recognition.face_encodings(rgb, boxes)[0]
                cap.release()
                cv2.destroyAllWindows()

                name = name_entry.get()
                roll = roll_entry.get()
                mobile = mobile_entry.get()
                gender = gender_var.get()
                student_class = class_entry.get()
                division = division_entry.get()

                if is_attendance_marked(name):
                    print(f"[INFO] {name} has already marked attendance today.")
                    return

                info = {
                    'name': name,
                    'roll': roll,
                    'mobile': mobile,
                    'gender': gender,
                    'class': student_class,
                    'division': division
                }
                save_face_encoding(encoding, info)

                # Save to student list if new
                if os.path.exists(STUDENT_LIST_FILE):
                    df = pd.read_csv(STUDENT_LIST_FILE)
                else:
                    df = pd.DataFrame(columns=['Name', 'Roll', 'Mobile', 'Gender', 'Class', 'Division'])

                if not ((df['Name'] == name) & (df['Roll'] == roll)).any():
                    df = pd.concat([df, pd.DataFrame([{
                        'Name': name,
                        'Roll': roll,
                        'Mobile': mobile,
                        'Gender': gender,
                        'Class': student_class,
                        'Division': division
                    }])], ignore_index=True)
                    df.to_csv(STUDENT_LIST_FILE, index=False)
                    print("[INFO] Student added to master list.")
                
                # Hide registration frame and show main options
                register_frame.pack_forget()
                show_main_options()

                return
            else:
                print("[WARNING] No face detected, try again.")

def mark_attendance(info):
    if is_attendance_marked(info['name']):
        print(f"[INFO] {info['name']} has already marked attendance today.")
        return
    
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    day = now.strftime("%A")

    new_row = {
        'Name': info['name'],
        'Roll': info['roll'],
        'Mobile': info['mobile'],
        'Gender': info['gender'],
        'Class': info['class'],
        'Division': info['division'],
        'Date': date,
        'Time': time_str,
        'Day': day
    }

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=['Name', 'Roll', 'Mobile', 'Gender', 'Class', 'Division', 'Date', 'Time', 'Day'])
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"[INFO] Attendance marked for {info['name']} at {time_str}")

def recognize_faces():
    print("[INFO] Starting recognition...")
    known_encodings, known_info = load_known_faces()
    cap = cv2.VideoCapture(1)

    last_recognized_roll = None
    last_recognized_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for enc, box in zip(encodings, boxes):
            matches = face_recognition.compare_faces(known_encodings, enc)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                info = known_info[best_match_index]
                current_time = time.time()
                if last_recognized_roll == info['roll'] and (current_time - last_recognized_time) < 3:
                    continue

                last_recognized_roll = info['roll']
                last_recognized_time = current_time

                mark_attendance(info)

                top, right, bottom, left = [v * 2 for v in box]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, info['name'], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Recognition - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def show_summary_gui(text_widget):
    text_widget.delete('1.0', tk.END)  # Clear previous summary

    if not os.path.exists(CSV_FILE):
        text_widget.insert(tk.END, "[INFO] No attendance data yet.\n")
        return

    df = pd.read_csv(CSV_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df['Date'] == today]

    if not os.path.exists(STUDENT_LIST_FILE):
        text_widget.insert(tk.END, "[INFO] Student list file is missing!\n")
        return

    student_df = pd.read_csv(STUDENT_LIST_FILE)
    all_students = student_df['Name'].tolist()
    present_students = today_df['Name'].tolist()
    absentees = [s for s in all_students if s not in present_students]
    absentee_df = student_df[student_df['Name'].isin(absentees)]

    # Save absentee CSV
    absentee_filename = f"absentees_{today}.csv"
    absentee_df.to_csv(absentee_filename, index=False)

    # NEW: Save present CSV
    present_filename = f"presents_{today}.csv"
    today_df.to_csv(present_filename, index=False)

    # Enable Download Buttons
    download_absent_button.config(state=tk.NORMAL, text=f"Download Absentees ({today})", command=lambda: os.startfile(absentee_filename))
    download_present_button.config(state=tk.NORMAL, text=f"Download Presents ({today})", command=lambda: os.startfile(present_filename))


    # Gender counts
    male_present = (today_df['Gender'] == 'M').sum()
    female_present = (today_df['Gender'] == 'F').sum()
    male_absent = (absentee_df['Gender'] == 'M').sum()
    female_absent = (absentee_df['Gender'] == 'F').sum()

    summary_text = f"\n[SUMMARY] {today}\n"
    summary_text += f"Total Present: {len(today_df)} | Males: {male_present}, Females: {female_present}\n"
    summary_text += f"Total Absent: {len(absentees)} | Males: {male_absent}, Females: {female_absent}\n\n"

    text_widget.insert(tk.END, summary_text)

    for _, row in absentee_df.iterrows():
        line = (
            f"Name: {row['Name']}, Gender: {row['Gender']}, "
            f"Class: {row['Class']}, Division: {row['Division']}, "
            f"Mobile: {row['Mobile']}, Roll: {row['Roll']}\n"
        )
        text_widget.insert(tk.END, line)

        # Send SMS
        mobile_number = str(row['Mobile'])
        if not mobile_number.startswith('+'):
            mobile_number = '+91' + mobile_number  # Adjust for your region
        message = f"Dear {row['Name']}, you were marked absent on {today}. Please ensure your presence next time."
        send_sms(mobile_number, message)

    # Enable Download Button
    download_button.config(state=tk.NORMAL, text=f"Download Absentees ({today})", command=lambda: os.startfile(absentee_filename))


def show_main_options():
    register_button.pack_forget()
    start_attendance_button.pack_forget()
    summary_button.pack_forget()
    
    register_button.pack(pady=10, fill='x', padx=20)
    start_attendance_button.pack(pady=10, fill='x', padx=20)
    summary_button.pack(pady=10, fill='x', padx=20)

def create_gui():
    global register_button, start_attendance_button, summary_button

    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("600x600")
    root.configure(bg='#F0F0F0')

    # Create a frame for registration input (initially hidden)
    register_frame = tk.Frame(root)

    name_label = tk.Label(register_frame, text="Name", font=("Arial", 12))
    name_label.grid(row=0, column=0, pady=5)
    name_entry = tk.Entry(register_frame, font=("Arial", 12))
    name_entry.grid(row=0, column=1, pady=5)

    roll_label = tk.Label(register_frame, text="Roll Number", font=("Arial", 12))
    roll_label.grid(row=1, column=0, pady=5)
    roll_entry = tk.Entry(register_frame, font=("Arial", 12))
    roll_entry.grid(row=1, column=1, pady=5)

    mobile_label = tk.Label(register_frame, text="Mobile Number", font=("Arial", 12))
    mobile_label.grid(row=2, column=0, pady=5)
    mobile_entry = tk.Entry(register_frame, font=("Arial", 12))
    mobile_entry.grid(row=2, column=1, pady=5)

    gender_label = tk.Label(register_frame, text="Gender", font=("Arial", 12))
    gender_label.grid(row=3, column=0, pady=5)
    gender_var = tk.StringVar()
    tk.Radiobutton(register_frame, text="Male", variable=gender_var, value="M").grid(row=3, column=1)
    tk.Radiobutton(register_frame, text="Female", variable=gender_var, value="F").grid(row=3, column=2)

    class_label = tk.Label(register_frame, text="Class", font=("Arial", 12))
    class_label.grid(row=4, column=0, pady=5)
    class_entry = tk.Entry(register_frame, font=("Arial", 12))
    class_entry.grid(row=4, column=1, pady=5)

    division_label = tk.Label(register_frame, text="Division", font=("Arial", 12))
    division_label.grid(row=5, column=0, pady=5)
    division_entry = tk.Entry(register_frame, font=("Arial", 12))
    division_entry.grid(row=5, column=1, pady=5)

    def handle_register():
        register_frame.pack_forget()  # Hide form after registration
        register_face(name_entry, roll_entry, mobile_entry, gender_var, class_entry, division_entry, register_frame)

    confirm_button = tk.Button(register_frame, text="Capture Face", font=("Arial", 14), bg='#4CAF50', fg='white', command=handle_register)
    confirm_button.grid(row=6, columnspan=2, pady=10)

    def show_register_form():
        register_button.pack_forget()
        start_attendance_button.pack_forget()
        summary_button.pack_forget()
        register_frame.pack(pady=20)

    def show_main_options():
        register_frame.pack_forget()
        register_button.pack(pady=10, fill='x', padx=20)
        start_attendance_button.pack(pady=10, fill='x', padx=20)
        summary_button.pack(pady=10, fill='x', padx=20)

    register_button = tk.Button(root, text="Register New Face", font=("Arial", 14), bg='#4CAF50', fg='white', command=show_register_form)
    start_attendance_button = tk.Button(root, text="Start Attendance", font=("Arial", 14), bg='#2196F3', fg='white', command=recognize_faces)
    summary_text_widget = scrolledtext.ScrolledText(root, width=70, height=15, wrap=tk.WORD)
    summary_button = tk.Button(root, text="Show Today's Summary", font=("Arial", 14), bg='#FFC107', fg='black', command=lambda: show_summary_gui(summary_text_widget))

    register_button.pack(pady=10, fill='x', padx=20)
    start_attendance_button.pack(pady=10, fill='x', padx=20)
    summary_button.pack(pady=10, fill='x', padx=20)
    summary_text_widget.pack(pady=20, padx=20)

    global download_absent_button, download_present_button

    # Absentee button
    download_absent_button = tk.Button(root, text="Download Absentees CSV", font=("Arial", 12), bg="#795548", fg="white", state=tk.DISABLED)
    download_absent_button.pack(pady=5, fill='x', padx=20)

    # Present button
    download_present_button = tk.Button(root, text="Download Presents CSV", font=("Arial", 12), bg="#4CAF50", fg="white", state=tk.DISABLED)
    download_present_button.pack(pady=5, fill='x', padx=20)

    

    root.mainloop()



if __name__ == "__main__":
    create_gui()
