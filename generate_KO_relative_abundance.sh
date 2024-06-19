#!/bin/bash
#
#cat << EOF > input.txt

input_name='input.txt'
#input_f='test1.txt'
total_files=$(cat $input_name | wc -l)
count1=0
while IFS= read -r line; do
    count1=$((count1 + 1))
    echo "$file  $output_1  $count1/$total_files"
    IFS=' ' read -ra data_array <<< "$line"

    if [ "${#data_array[@]}" -ge 4 ]; then
        input_original_sra="${data_array[0]}"
        output_sra="${data_array[1]}"
        input_fastq="${data_array[1]}"        
        out_qc_fastq="${data_array[2]}"
        out_qc_rm_fastq="${data_array[3]}"
        out_ko="${data_array[4]}"
        echo "input_original_sra: $input_original_sra"
        echo "output_sra: $output_sra"
        echo "out_qc_fastq: $out_qc_fastq"
        echo "out_qc_rm_fastq: $out_qc_fastq"
        echo "out_ko: $out_ko"
    fi
    if [ -d "$output_sra" ]; then
        echo "file has been created: $output_sra"
    else
        mkdir -p "$output_sra"
        echo "success finished: $output_sra"
    fi

    if [ -d "$out_qc_fastq" ]; then
        echo "file has been created: $out_qc_fastq"
    else
        mkdir -p "$out_qc_fastq"
        echo "success finished: $out_qc_fastq"
    fi

     if [ -d "$out_qc_rm_fastq" ]; then
        echo "file has been created: $out_qc_rm_fastq"
    else
        mkdir -p "$out_qc_rm_fastq"
        echo "success finished: $out_qc_rm_fastq"
    fi   

    echo "$input_original_sra"
    total_files=$(ls $input_original_sra | wc -l)
    count=0
    file_list=$(ls "$input_original_sra" | sort | uniq)
    #while IFS= read -r file; do
    for file in $file_list; do
        input_="${input_original_sra}/${file}/${file}.*"

        output_1="${output_sra}/${file}_1.fastq"
        output_2="${output_sra}/${file}_2.fastq"

        count=$((count + 1))
        echo "$file  $output_1  $count/$total_files"

        if [ -f "$output_1" ]; then
            echo "Output file $output_1 already exists. Skipping..."
        else
            fastq-dump $input_ --split-3 -O $output_sra
            if [ -f "$output_1" ]; then
                #rm -rf $input_
                echo "finished ${output_1}"
            fi

        fi
    done
    #QC
    #total_files=$(cat $input_original_sra | wc -l)
    echo "$input_fastq"    
    count=0
    #while IFS= read -r file; do
    for file in $file_list; do
        input_1="${input_fastq}/${file}_1.fastq"
        input_2="${input_fastq}/${file}_2.fastq"

        output_1="${out_qc_fastq}/${file}_1.fastq"
        output_2="${out_qc_fastq}/${file}_2.fastq"

        count=$((count + 1))
        echo "$file  $output_1  $count/$total_files"

        if [ -f "$output_1" ]; then
            echo "Output file $output_1 already exists. Skipping..."
        else
            fastp -i $input_1 -I $input_2 -o $output_1 -O $output_2 -w 16
            if [ -f "$output_1" ]; then
                rm -rf $input_1
                rm -rf $input_2
            fi

        fi
    done

    echo "$out_qc_fastq"    
    count=0
    #while IFS= read -r file; do
    for file in $file_list; do
        input_1="${out_qc_fastq}/${file}_1.fastq"
        input_2="${out_qc_fastq}/${file}_2.fastq"

        output_1="${out_qc_rm_fastq}/${file}_1.fastq"
        output_2="${out_qc_rm_fastq}/${file}_2.fastq"

        output_="${out_qc_rm_fastq}/${file}"

        count=$((count + 1))
        echo "$file  $output_  $count/$total_files"

        if [ -f "$output_1" ]; then
            echo "Output file $output_ already exists. Skipping..."
        else
            #bowtie2 -x /hde/GRCh38_noalt_as/GRCh38_noalt_as -1 $input_1 -2 $input_2 -p 48 --un-conc $output_
            bowtie2 -x /hde/GRCh38_noalt_as/GRCh38_noalt_as -1 $input_1 -2 $input_2 -p 48 -S ${output_}.sam
            samtools view -@ 32 -bS ${output_}.sam > ${output_}.bam
            samtools view -@ 32 -bf 12 ${output_}.bam > ${out_qc_rm_fastq}/unmapped.bam
            #samtools sort -@ 32 ${out_qc_rm_fastq}/unmapped.bam -o ${out_qc_rm_fastq}/sorted.bam
            samtools fastq -@ 32 ${out_qc_rm_fastq}/unmapped.bam -1 $output_1 -2 $output_2       
            if [ -f "$output_1" ]; then
                echo "Output file ${output_1} already exists. deleting..."
                rm -rf ${out_qc_rm_fastq}/*.sam
                rm -rf ${out_qc_rm_fastq}/*.bam
                rm -rf $input_1
                rm -rf $input_2
            fi

        fi
    done



    diting.py -r $out_qc_rm_fastq -o $out_ko -n 24

    if [ -f "${out_ko}/KEGG_annotation/ko_abundance_among_samples.tab" ]; then
        rm -rf ${out_qc_rm_fastq}/*
        echo "delete qc_rm: ${out_qc_rm_fastq}"
    fi
done < $input_name







