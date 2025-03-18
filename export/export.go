package export

import (
	"bytes"
	"fmt"
	"goai/neuralnet"
	"goai/samples"
	"goai/utils"
	"os"
)

func NetworkResults(net *neuralnet.Net, smpls []samples.Sample) {
	run := neuralnet.NewNetRun(net)
	inp_buffer := bytes.NewBuffer([]byte{})
	out_buffer := bytes.NewBuffer([]byte{})
	exp_out_buffer := bytes.NewBuffer([]byte{})

	for sample_index := range smpls {
		run.ForwardAll(smpls[sample_index].Input)
		inp_buffer.WriteString(fmt.Sprintf("%f,", smpls[sample_index].Input[0]))
		exp_out_buffer.WriteString(fmt.Sprintf("%f,", smpls[sample_index].Expected_output[0]))
		out_buffer.WriteString(fmt.Sprintf("%f,", run.GetFinalSigmoidedOutput()[0]))
	}

	inp_str := inp_buffer.String()
	out_string := out_buffer.String()
	exp_out_str := exp_out_buffer.String()

	file, err := os.Create("results.csv")
	utils.Expect(err)
	defer file.Close()
	file.WriteString(inp_str[:len(inp_str)-1])
	file.WriteString("\n")
	file.WriteString(out_string[:len(out_string)-1])
	file.WriteString("\n")
	file.WriteString(exp_out_str[:len(exp_out_str)-1])
}
