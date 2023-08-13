import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { memo, useCallback } from 'react';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaTrash } from 'react-icons/fa';
import { useAppDispatch } from 'app/store/storeHooks';
import { workflowExposedFieldRemoved } from 'features/nodes/store/nodesSlice';

type Props = {
  nodeData: InvocationNodeData;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue;
  fieldTemplate: InputFieldTemplate;
};

const LinearViewField = ({
  nodeData,
  nodeTemplate,
  field,
  fieldTemplate,
}: Props) => {
  const dispatch = useAppDispatch();
  const handleRemoveField = useCallback(() => {
    dispatch(
      workflowExposedFieldRemoved({
        nodeId: nodeData.id,
        fieldName: field.name,
      })
    );
  }, [dispatch, field.name, nodeData.id]);

  return (
    <Flex
      layerStyle="second"
      sx={{ position: 'relative', borderRadius: 'base', w: 'full', p: 2 }}
    >
      <FormControl as={Flex} sx={{ flexDir: 'column', gap: 1 }}>
        <Tooltip
          label={
            <FieldTooltipContent
              nodeData={nodeData}
              nodeTemplate={nodeTemplate}
              field={field}
              fieldTemplate={fieldTemplate}
            />
          }
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
          shouldWrapChildren
          hasArrow
        >
          <FormLabel
            sx={{
              mb: 0,
            }}
          >
            {field.label || fieldTemplate.title} (
            {nodeData.label || nodeTemplate.title})
          </FormLabel>
        </Tooltip>
        <InputFieldRenderer
          nodeData={nodeData}
          nodeTemplate={nodeTemplate}
          field={field}
          fieldTemplate={fieldTemplate}
        />
      </FormControl>
      <IAIIconButton
        onClick={handleRemoveField}
        aria-label="Remove"
        label="Remove"
        icon={<FaTrash />}
        size="xs"
        variant="ghost"
        sx={{
          position: 'absolute',
          top: 1,
          insetInlineEnd: 1,
        }}
      />
    </Flex>
  );
};

export default memo(LinearViewField);
