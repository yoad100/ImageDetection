package com.example.imagedetection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.imagedetection.ml.MobilenetV110224Quant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.TensorFlowLite;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private static final int RESULT_PIC = 0;
    ImageView img;
    ArrayList<String> arr;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        arr = new ArrayList<String>();
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt"),"UTF-8"));
            String mline;
            while ((mline = reader.readLine())!=null){
                arr.add(mline);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void selectFunc(View view) {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, RESULT_PIC);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

         img = findViewById(R.id.ImageSelected);
         if(resultCode == RESULT_OK){
            final Uri imageUri = data.getData();
            try {
                final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                img.setImageBitmap(selectedImage);
            }catch (FileNotFoundException e){
                e.printStackTrace();
            }

         }else{
             Toast.makeText(this,"you havent picked image",Toast.LENGTH_LONG).show();
         }
    }

    public void detectFunc(View view) {
        TextView textView =findViewById( R.id.probability1);
        TextView textView2 =findViewById( R.id.probability2);
        TextView textView3 =findViewById( R.id.probability3);
        try {
            Bitmap bm = ((BitmapDrawable) img.getDrawable()).getBitmap();
            Bitmap resize = Bitmap.createScaledBitmap(bm,224,224,true);

            TensorImage selectedImage = TensorImage.fromBitmap(resize);
            ByteBuffer byteBuffer = selectedImage.getBuffer();

            MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] result = findMax(outputFeature0.getFloatArray());
            textView.setText(arr.get((int) result[3]) + "        "+result[0]);
            textView2.setText(arr.get((int) result[4]) + "        "+result[1]);
            textView3.setText(arr.get((int) result[5]) + "        "+result[2]);
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
    public float[] findMax (float[] probabilityArr){
        int index1 = 0;
        int index2 = 0;
        int index3 = 0;

        float bestProbability1=0;
        float bestProbability2=0;
        float bestProbability3=0;
        for (int i = 0 ; i<probabilityArr.length; i++){
           if(probabilityArr[i]>bestProbability1){
               bestProbability1=probabilityArr[i];
               index1=i;
           }

           else if (probabilityArr[i]>bestProbability2){
               bestProbability2=probabilityArr[i];
               index2=i;
            }

           else if (probabilityArr[i]>bestProbability3){
               bestProbability3=probabilityArr[i];
               index3=i;
            }


        }
        return new float[]{bestProbability1,bestProbability2,bestProbability3,index1,index2,index3};
    }
}