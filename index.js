import { PNG } from "pngjs/browser"
import str from 'string-to-stream'
import { keep, time } from "@tensorflow/tfjs"
var jpeg = require('jpeg-js')
var tf = require('@tensorflow/tfjs')
var now = require('performance-now')


addEventListener('fetch', event => {
  event.respondWith(handleRequest(event))
})

async function handleRequest({ request }) {
  try {
    if (request.method === 'GET') {
      const res = new Response(`
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="UTF-8">
          <title>MobileNetv2</title>
        </head>
        <body>
          Upload an image (JPEG or PNG):
          <input type="file" onchange="post()"/>

          <script>
            function post() {
              var file = document.querySelector('input[type=file]').files[0];
              var reader = new FileReader();

              reader.addEventListener('load', function () {
                let imgEl = document.getElementById("img_el");
                imgEl.src = reader.result;
                imgEl.setAttribute("width", 28);
                imgEl.setAttribute("height", 28);
                document.body.append(imgEl);
                
                fetch(location.href, {
                  method: "POST",
                  body: reader.result
                })
                  .then(res => res.json())
                  .then(pred => {
                    document.getElementById("pred").innerText = "Prediction: " + pred.label
                    document.getElementById("model-time").innerText = "Model Load time: " + pred["model-time"]
                    document.getElementById("inf-time").innerText = "Inference time: " + pred["inf-time"]
                  });
              }, false);

              if (file) {
                reader.readAsDataURL(file);
              }
            }
          </script>
          <img id="img_el"></img>
          <p id ="pred"></p>
          <p id="model-time"></p>
          <p id="inf-time"></p>
        </body>
      </html>
    `)
      res.headers.set('content-type', 'text/html')
      return res
    }

    if (request.method === 'POST') {
      var model_start = now()
      const model = await componentDidMount()
      var model_end = now()
      const base64String = await request.text()
      const [dtype, data] = base64String.split(",")
      if (dtype === 'data:image/jpeg;base64') {
        const rawImageData = jpeg.decode(base64ToArrayBuffer(data))
        const tensor = bufferToTensor(rawImageData.data)
        const pred = await classifyImage(model, tensor)
        var infend = now()
        var modelload = model_end-model_start
        var infdiff = infend-model_end
        return new Response(JSON.stringify({"label": pred, "model-time": modelload.toString()+"ms", "inf-time": infdiff.toString()+"ms"}), { status: 200 })
      }
      if (dtype === 'data:image/png;base64') {
        var resp = await new Promise((resolve, reject) => {
          try {
            str(base64ToArrayBuffer(data))
              .pipe(
                new PNG({
                  filterType: 4
                })
              )
              .on('parsed', async function() {
                const tensor = bufferToTensor(this.data)
                const pred = await classifyImage(model, tensor);
                var infend = now()
                var modelload = model_end-model_start
                var infdiff = infend-model_end
                resolve(new Response(JSON.stringify({"label": pred, "model-time": modelload.toString()+"ms", "inf-time": infdiff.toString()+"ms"}), { status: 200 }))
              })
          } catch (e) {
            console.error(e)
          }
        })
        return resp
      }
    }

    return new Response('not found', { status: 404 })
  } catch (err) {
    console.log(err)
    return new Response(err, { status: 500 })
  }
}

async function componentDidMount() {
  await tf.ready()
  const model = await tf.loadLayersModel("https://storage.googleapis.com/lenet/model.json").catch(e => console.error(e))
  return model
}

async function classifyImage(model, inputTensor) {
  const prediction = await model.predict(inputTensor).array()
  console.log(prediction)
  return prediction[0].indexOf(Math.max(...prediction[0]));
}

function base64ToArrayBuffer(base64) {
  var binary_string = atob(base64)
  var len = binary_string.length
  var bytes = new Uint8Array(len)
  for (var i = 0; i < len; i++) {
    bytes[i] = binary_string.charCodeAt(i)
  }
  return bytes.buffer
}

function bufferToTensor(dataArray) {
  var tensor = tf.tensor(dataArray, [1,28,28,4], 'int32')
  var subTensor = tf.slice(tensor, [0,0,0,0], [1,28,28,1])
  return subTensor
}