#!/bin/bash

# This is an example script how you can send submissions to the robot in a
# somewhat automated way.  The basic idea is the following:
#
#  1. submit job to robot
#  2. wait until job is finished
#  3. download relevant data from the run
#  4. run some algorithm on the data (e.g. to compute a new policy)
#  5. update the policy parameters in the git repository and push the
#     changes
#  6. goto 1 unless some termination criterion is fulfilled
#
# You can use the script mostly as is, you would just need to add your code for
# processing the data and pushing new parameters to git below.  You can also
# adapt any part of the script to your needs.  However, to avoid overloading
# our system, you MUST NOT poll the server at a higher rate than once per
# minute to see the status of the job!


# expects output directory (to save downloaded files) as argument
if (( $# != 1 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <output_directory>"
    exit 1
fi

output_directory="$1"

if ! [ -d "${output_directory}" ]
then
    echo "${output_directory} does not exist or is not a directory"
    exit 1
fi

# prompt for username and password (to avoid having user credentials in the
# bash history)
read -p "Username: " username
read -sp "Password: " password
# there is no automatic new line after the password prompt
echo


# URL to the webserver at which the recorded data can be downloaded
base_url=https://robots.real-robot-challenge.com/output/${username}/data


# Check if the file/path given as argument exists in the "data" directory of
# the user
function curl_check_if_exists()
{
    local filename=$1

    http_code=$(curl -sI --user ${username}:${password} -o /dev/null -w "%{http_code}" ${base_url}/${filename})

    case "$http_code" in
        200|301) return 0;;
        *) return 1;;
    esac
}


# Send 30 submissions in a loop.  You may want to replace the loop condition
# with something more useful.
for (( i=0; i<30; i++))
do
    echo "Submit job"
    submit_result=$(ssh -T ${username}@subrob.is.localnet <<<submit)
    job_id=$(echo ${submit_result} | grep -oP 'job\(s\) submitted to cluster \K[0-9]+')
    if [ $? -ne 0 ]
    then
        echo "Failed to submit job.  Output:"
        echo "${submit_result}"
        exit 1
    fi
    echo "Submitted job with ID ${job_id}"

    echo "Wait for job to be started"
    job_started=0
    # wait for the job to start (abort if it did not start after half an hour)
    for (( i=0; i<30; i++))
    do
        # Do not poll more often than once per minute!
        sleep 60

        # wait for the job-specific output directory
        if curl_check_if_exists ${job_id}
        then
            job_started=1
            break
        fi
        date
    done

    if (( ${job_started} == 0 ))
    then
        echo "Job did not start."
        exit 1
    fi

    echo "Job is running.  Wait until it is finished"
    # if the job did not finish 10 minutes after it started, something is
    # wrong, abort in this case
    job_finished=0
    for (( i=0; i<15; i++))
    do
        # Do not poll more often than once per minute!
        sleep 60

        # report.json is explicitly generated last of all files, so once this
        # exists, the job has finished
        if curl_check_if_exists ${job_id}/report.json
        then
            job_finished=1
            break
        fi
        date
    done

    # create directory for this job
    job_dir="${output_directory}/${job_id}"
    mkdir "${job_dir}"

    if (( ${job_finished} == 0 ))
    then
        echo "Job did not finish in time."
        exit 1
    fi

    echo "Job ${job_id} finished."

    echo "Download data to ${job_dir}"

    # Download data.  Here only the report file is downloaded as example.  Add
    # equivalent commands for other files as needed.
    curl --user ${username}:${password} -o "${job_dir}/report.json" ${base_url}/${job_id}/report.json

    # if there was a problem with the backend, download its output and exit
    if grep -q "true" "${job_dir}/report.json"
    then
        echo "ERROR: Backend failed!  Download backend output and stop"
        curl --user ${username}:${password} -o "${job_dir}/stdout.txt" ${base_url}/../stdout.txt
        curl --user ${username}:${password} -o "${job_dir}/stderr.txt" ${base_url}/../stderr.txt

        exit 1
    fi


    # TODO: process data in some way
    #
    # Here you can now process the data recorded during the run.  You can, for
    # example, train some model based on the recorded data and update some file
    # with parameters that would be used during the next run on the robot.
    # Remember that any updates need to be pushed to your git repository before
    # submitting the next job to the robot.
    #
    # Below is an example how that could look like.  This assumes that you are
    # running this script inside your git repository and that you have a file
    # 'train.py' which computes new parameters, stored in a file
    # 'parameters.json' which will be read by the 'run' script when executing
    # it on the robot.  You can, of course have any other structure, just
    # adjust the commands accordingly.
    #
    #     ./train.py --input ${job_dir}/robot_data.csv --output ./parameters.json
    #     git add ./parameters.json
    #     git commit -am "Update parameters"
    #     git push
    #
    # NOTE: The robot cluster system needs up to 60 seconds until the next job
    # can be submitted after the previous one finished.  So depending how long
    # your data processing code above (if existent) takes, you may need to add
    # a sleep here.
    sleep 60

    echo
    echo "============================================================"
    echo

done
