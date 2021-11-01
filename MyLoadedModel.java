package com.xx.examples;

import java.io.IOException;
import java.nio.file.*;
import java.lang.*;
import java.net.URL;

import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.channels.FileChannel;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.*;
import ai.djl.modality.cv.transform.*;
import ai.djl.modality.cv.translator.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;
import ai.djl.ndarray.*;

import java.util.Arrays;

public class MyLoadedModel {

    public static Predictor<Float[], Float[]> predictor;
    static {
        try {
            loadModel();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static final class FloatToFloatTranslator implements Translator<Float[], Float[]> {

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }

        @Override
        public Float[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
            NDArray tempArray = ndList.get(0);
            float[] ret = tempArray.toFloatArray();
            Float[] ret1 = new Float[ret.length];
            for (int i = 0, L = ret.length; i<L; i++) {
                ret1[i] = new Float(ret[i]);
            }
            return ret1;
        }

        @Override
        public NDList processInput(TranslatorContext translatorContext, Float[] features) throws Exception {
            float[] in1 = new float[features.length];
            for (int i = 0, L = features.length; i<L; i++) {
                in1[i] = features[i];
            }
            NDManager manager = translatorContext.getNDManager();
            return new NDList(manager.create(in1));
        }
    }

    public static void loadModel() throws Exception {
        Criteria<Float[], Float[]> criteria = Criteria.builder()
                .setTypes(Float[].class, Float[].class)
                .optTranslator(new FloatToFloatTranslator())
                .optModelUrls("jar:///test_model.tgz")
                .optModelName("model_tc_1")
                .build();

        ZooModel<Float[], Float[]> model = ModelZoo.loadModel(criteria);
        predictor = model.newPredictor();
    }

    public static void main(String[] args) {
        Predictor<Float[], Float[]> predictor = MyLoadedModel.predictor;
    }
}
