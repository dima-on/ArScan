import time
from flask import Flask, render_template, request, jsonify
import WorkFile



app = Flask(__name__)
def testIMG(img, inT, inD, offsetY, offset_Down, with_Down, with_Top, offsetX_Top, offsetDownX, t):
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    return WorkFile.resImage(img, inT, inD, offsetY, offset_Down, with_Down, with_Top, offsetX_Top, offsetDownX, t)


@app.route('/')
def index():
    # Вызовите функции из вашей программы
    WorkFile.BeginFun()
    return render_template('GUI/index.html')


@app.route('/lick', methods=['POST'])
def lick():
    print("test")
    # Обработка загрузки фото
    uploaded_file = request.files['photo']
    offset_Top = request.form.get('offset')
    Data = request.form.get('photoData')
    offsetX_Top = request.form.get('offsetX')
    with_Top = request.form.get('UpWith')
    TopIndex = request.form.get('TopIndex')

    offset_Down = request.form.get('offsetDown')
    offset_DownX = request.form.get('offsetXDown')
    with_Down = request.form.get('DownWith')
    DownIndex = request.form.get('DownIndex')

    print("end")


    if uploaded_file.filename != '':


        # Сохраняем загруженное фото в папке uploads
        upload_folder = 'static/Input/'
        uploaded_file.save(upload_folder + uploaded_file.filename)

        path = upload_folder + uploaded_file.filename
        print(DownIndex)
        im = testIMG(path, int(TopIndex), int(DownIndex), int(offset_Top), int(offset_Down), int(with_Down), int(with_Top), int(offsetX_Top), int(offset_DownX), time.time())


    return jsonify({'result_image': im})


@app.route('/StartProgram', methods=['POST'])
def StartAll():
    Top_Image_Path = WorkFile.Top_Image_Path
    Down_Image_Path = WorkFile.Down_Image_Path
    return jsonify({'result_image': Top_Image_Path, 'result_imageD': Down_Image_Path})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
